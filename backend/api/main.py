from contextlib import asynccontextmanager
import logging
import os
import time

import psutil
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend.db.session import init_db, engine
from backend.api.routes import jobs, models, datasets, exports, asr, eval as eval_routes, chat, conversations, prompt_profiles, sandbox, provider_client, inference, settings as settings_routes

# ── Logging ───────────────────────────────────────────────────────────────────
_log_level = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("forge.api")


# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialised")
    yield
    logger.info("API shutting down")


app = FastAPI(title="Forge Training API", version="1.0.0", lifespan=lifespan)

# ── CORS ──────────────────────────────────────────────────────────────────────
_raw_origins = os.getenv("FRONTEND_URL", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    logger.info("%s %s → %d  (%.0f ms)", request.method, request.url.path, response.status_code, ms)
    return response


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(jobs.router)
app.include_router(models.router)
app.include_router(datasets.router)
app.include_router(exports.router)
app.include_router(asr.router)
app.include_router(eval_routes.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(prompt_profiles.router)
app.include_router(sandbox.router)
app.include_router(provider_client.router)
app.include_router(inference.router)
app.include_router(settings_routes.router)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    redis_ok = False
    try:
        import redis as _redis
        _r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), socket_connect_timeout=2)
        _r.ping()
        redis_ok = True
    except Exception:
        pass

    status = "ok" if (db_ok and redis_ok) else "degraded"
    return {"status": status, "db": db_ok, "redis": redis_ok}


def _get_gpu_info() -> list:
    """Query real GPU memory via pynvml (nvidia-smi style), fall back to torch."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpus.append({
                "index": i,
                "name": name if isinstance(name, str) else name.decode(),
                "total_mb": mem.total // (1024 ** 2),
                "used_mb": mem.used // (1024 ** 2),
                "free_mb": mem.free // (1024 ** 2),
                "utilization_percent": util.gpu,
            })
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        pass

    # Fallback: torch (only sees memory allocated by this process)
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            total = props.total_memory
            gpus.append({
                "index": i,
                "name": props.name,
                "total_mb": total // (1024 ** 2),
                "used_mb": allocated // (1024 ** 2),
                "free_mb": (total - allocated) // (1024 ** 2),
                "utilization_percent": None,
            })
        return gpus
    except Exception:
        return []


# ── System stats ──────────────────────────────────────────────────────────────
@app.get("/api/system")
def system_stats():
    gpu_info = _get_gpu_info()

    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        pass

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(".")
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_total_mb": mem.total // (1024 ** 2),
        "ram_used_mb": mem.used // (1024 ** 2),
        "disk_total_gb": disk.total // (1024 ** 3),
        "disk_used_gb": disk.used // (1024 ** 3),
        "gpu": gpu_info,
        "cuda_available": cuda_available,
    }
