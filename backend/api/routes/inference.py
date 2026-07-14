"""
inference.py — on-demand inference service using merged/downloaded models only.

Rules:
  - Only merged exports (EXPORTS_DIR) or downloaded base models (ModelEntry.is_downloaded == "true")
    are eligible. Attempting to load any other path returns 400.
  - One model loaded at a time (single GPU assumption). Use POST /api/inference/unload to release.
  - POST /api/inference/fill-dataset pulls a text dataset from DataSupportTool's provider API,
    runs the loaded model on every record, and returns the filled records (output field populated).
"""
from __future__ import annotations
import asyncio
import gc
import os
from pathlib import Path
from threading import Lock, Thread
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.core.settings import get_provider_api_key, get_provider_base_url
from backend.db.models import ModelEntry

router = APIRouter(prefix="/api/inference", tags=["inference"])

EXPORTS_DIR = os.getenv("EXPORTS_DIR", "./exports")

# ── Singleton model state ──────────────────────────────────────────────────────

_state: dict = {
    "model":      None,
    "tokenizer":  None,
    "model_path": None,
    "status":     "unloaded",  # unloaded | loading | ready | error
    "error":      None,
}
_lock = Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_merged_export(path: str) -> bool:
    exports_real = os.path.realpath(EXPORTS_DIR)
    candidate = os.path.realpath(path)
    return candidate.startswith(exports_real + os.sep) or candidate == exports_real


def _is_downloaded_base(path: str, db: Session) -> bool:
    m = db.query(ModelEntry).filter(
        ModelEntry.is_downloaded == "true",
    ).filter(
        (ModelEntry.local_path == path) | (ModelEntry.hf_repo == path)
    ).first()
    return m is not None


def _eligible(path: str, db: Session) -> bool:
    return _is_merged_export(path) or _is_downloaded_base(path, db)


def _unload_current():
    with _lock:
        _state["model"] = None
        _state["tokenizer"] = None
        _state["model_path"] = None
        _state["status"] = "unloaded"
        _state["error"] = None
    try:
        import torch
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass


def _do_load(model_path: str, quantization: Optional[str]):
    from backend.core.model.loader import load_model, load_tokenizer

    with _lock:
        _state["status"] = "loading"
        _state["error"] = None
    try:
        tokenizer = load_tokenizer(model_path)
        model = load_model(model_path, quantization=quantization)
        with _lock:
            _state["tokenizer"] = tokenizer
            _state["model"] = model
            _state["model_path"] = model_path
            _state["status"] = "ready"
    except Exception as exc:
        with _lock:
            _state["status"] = "error"
            _state["error"] = str(exc)


def _do_generate(prompt: str, system_prompt: Optional[str], params: dict) -> str:
    import torch

    with _lock:
        model = _state["model"]
        tokenizer = _state["tokenizer"]
    if model is None:
        raise RuntimeError("No model loaded")

    messages: list = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = (f"System: {system_prompt}\n\n" if system_prompt else "") + f"Human: {prompt}\n\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            do_sample=params["temperature"] > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── List eligible models ───────────────────────────────────────────────────────

@router.get("/models", summary="List models eligible for inference (merged exports + downloaded base models)")
def list_inference_models(db: Session = Depends(get_db)):
    merged: list = []
    exports_path = Path(EXPORTS_DIR)
    if exports_path.exists():
        for p in sorted(exports_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_dir():
                size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 2)
                merged.append({
                    "type": "merged",
                    "name": p.name,
                    "path": str(p),
                    "size_mb": round(size_mb, 1),
                })

    base_rows = db.query(ModelEntry).filter(ModelEntry.is_downloaded == "true").all()
    base: list = [
        {
            "type": "base",
            "id": m.id,
            "name": m.name,
            "hf_repo": m.hf_repo,
            "path": m.local_path or m.hf_repo,
            "template": m.template,
            "version": m.version,
        }
        for m in base_rows
    ]

    return {"merged": merged, "base": base}


# ── Status ─────────────────────────────────────────────────────────────────────

@router.get("/status", summary="Current inference model status")
def inference_status():
    with _lock:
        return {
            "status":     _state["status"],
            "model_path": _state["model_path"],
            "error":      _state["error"],
        }


# ── Load ───────────────────────────────────────────────────────────────────────

class LoadRequest(BaseModel):
    model_path: str
    quantization: Optional[str] = None  # none | 4bit | 8bit


@router.post("/load", summary="Load a merged or downloaded model for inference")
def load_inference_model(body: LoadRequest, db: Session = Depends(get_db)):
    if not _eligible(body.model_path, db):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only merged exports (from the Exports page) or registered downloaded "
                "base models may be used for inference. "
                f"'{body.model_path}' does not qualify."
            ),
        )

    with _lock:
        if _state["status"] == "loading":
            raise HTTPException(status_code=409, detail="A model is already loading — wait for it to finish")

    _unload_current()
    Thread(target=_do_load, args=(body.model_path, body.quantization), daemon=True).start()
    return {"message": "Loading started", "model_path": body.model_path}


# ── Unload ─────────────────────────────────────────────────────────────────────

@router.post("/unload", summary="Unload the current inference model and free GPU memory")
def unload_inference_model():
    _unload_current()
    return {"message": "Model unloaded"}


# ── Single generate ────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


def _assert_ready():
    with _lock:
        status = _state["status"]
    if status != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Inference model not ready (status: {status}). POST /api/inference/load first.",
        )


@router.post("/generate", summary="Run inference on a single prompt using the loaded model")
async def generate(body: GenerateRequest):
    _assert_ready()
    params = {k: getattr(body, k) for k in ("max_new_tokens", "temperature", "top_p", "repetition_penalty")}
    try:
        output = await asyncio.to_thread(_do_generate, body.prompt, body.system_prompt, params)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")
    return {"prompt": body.prompt, "output": output}


# ── Batch generate ─────────────────────────────────────────────────────────────

class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    system_prompt: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


@router.post("/generate/batch", summary="Run inference on multiple prompts sequentially")
async def generate_batch(body: BatchGenerateRequest):
    _assert_ready()
    if not body.prompts:
        raise HTTPException(status_code=422, detail="prompts list is empty")
    params = {k: getattr(body, k) for k in ("max_new_tokens", "temperature", "top_p", "repetition_penalty")}

    def _run():
        return [_do_generate(p, body.system_prompt, params) for p in body.prompts]

    try:
        outputs = await asyncio.to_thread(_run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation error: {e}")
    return {"results": [{"prompt": p, "output": o} for p, o in zip(body.prompts, outputs)]}


# ── Fill dataset from DataSupportTool ─────────────────────────────────────────

class FillDatasetRequest(BaseModel):
    dataset_id: int
    format: str = "sharegpt"          # alpaca | sharegpt (gemma not fillable)
    system_prompt: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


def _extract_prompt(rec: dict, fmt: str) -> str:
    if fmt == "alpaca":
        ctx = rec.get("input", "")
        q   = rec.get("instruction", "")
        return f"{ctx}\n\n{q}".strip() if ctx else q
    if fmt == "sharegpt":
        convs = rec.get("conversations", [])
        human = next((c["value"] for c in convs if c.get("from") == "human"), "")
        return human
    return str(rec)


def _fill_record(rec: dict, fmt: str, output: str) -> dict:
    if fmt == "alpaca":
        return {**rec, "output": output}
    if fmt == "sharegpt":
        convs = [c for c in rec.get("conversations", []) if c.get("from") != "gpt"]
        return {"conversations": [*convs, {"from": "gpt", "value": output}]}
    return {**rec, "output": output}


@router.post(
    "/fill-dataset",
    summary="Pull a text dataset from DataSupportTool and fill the output field using the loaded model",
)
async def fill_dataset(body: FillDatasetRequest, db: Session = Depends(get_db)):
    if body.format not in ("alpaca", "sharegpt"):
        raise HTTPException(status_code=422, detail="format must be 'alpaca' or 'sharegpt'")

    _assert_ready()

    base = get_provider_base_url(db)
    key  = get_provider_api_key(db)
    if not base or not key:
        raise HTTPException(status_code=503, detail="Provider endpoint / API key not configured — set them on the Settings page")

    async with httpx.AsyncClient(base_url=base, headers={"X-API-Key": key}, timeout=60.0) as c:
        r = await c.get(f"/api/provider/text/{body.dataset_id}/export", params={"format": body.format})
    if r.status_code == 401:
        raise HTTPException(status_code=401, detail="DataSupportTool: invalid API key")
    if r.status_code == 503:
        raise HTTPException(status_code=503, detail="DataSupportTool: provider API is disabled")
    if r.status_code == 404:
        raise HTTPException(status_code=404, detail="DataSupportTool: dataset not found")
    r.raise_for_status()
    payload = r.json()

    records: list = payload.get("records", [])
    if not records:
        raise HTTPException(status_code=422, detail="No verified records in this dataset yet")

    params = {k: getattr(body, k) for k in ("max_new_tokens", "temperature", "top_p", "repetition_penalty")}

    def _run_all():
        results = []
        for rec in records:
            prompt = _extract_prompt(rec, body.format)
            output = _do_generate(prompt, body.system_prompt, params) if prompt else ""
            results.append(_fill_record(rec, body.format, output))
        return results

    try:
        filled = await asyncio.to_thread(_run_all)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "dataset_id":   body.dataset_id,
        "dataset_name": payload.get("dataset_name"),
        "format":       body.format,
        "model_path":   _state["model_path"],
        "record_count": len(filled),
        "records":      filled,
    }
