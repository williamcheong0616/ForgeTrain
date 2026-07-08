"""
provider_client.py — pulls data FROM DataSupportTool's read-only provider API
and optionally imports it into RojBot's local dataset registry.

Env vars required:
  PROVIDER_BASE_URL  — e.g. http://datasupporttool:8000
  PROVIDER_API_KEY   — shared secret from DataSupportTool's .env
"""
from __future__ import annotations
import asyncio
import json
import os
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.db.models import Dataset

router = APIRouter(prefix="/api/provider", tags=["provider"])

DATASETS_DIR = os.getenv("DATASETS_DIR", "./datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)


def _base_url() -> str:
    url = os.getenv("PROVIDER_BASE_URL", "").rstrip("/")
    if not url:
        raise HTTPException(status_code=503, detail="PROVIDER_BASE_URL is not configured")
    return url


def _api_key() -> str:
    key = os.getenv("PROVIDER_API_KEY", "")
    if not key:
        raise HTTPException(status_code=503, detail="PROVIDER_API_KEY is not configured")
    return key


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=_base_url(),
        headers={"X-API-Key": _api_key()},
        timeout=60.0,
    )


def _check(r: httpx.Response) -> None:
    if r.status_code == 401:
        raise HTTPException(status_code=401, detail="DataSupportTool: invalid or missing API key")
    if r.status_code == 503:
        raise HTTPException(status_code=503, detail="DataSupportTool: provider API is disabled on that server")
    if r.status_code == 404:
        raise HTTPException(status_code=404, detail="DataSupportTool: resource not found")
    r.raise_for_status()


# ── Discovery ──────────────────────────────────────────────────────────────────

@router.get("/datasets", summary="List all datasets available on DataSupportTool")
async def list_provider_datasets():
    async with _client() as c:
        r = await c.get("/api/provider/datasets")
    _check(r)
    return r.json()


# ── Text dataset ───────────────────────────────────────────────────────────────

@router.get("/text/{dataset_id}/export", summary="Proxy a text export from DataSupportTool (no import)")
async def proxy_text_export(
    dataset_id: int,
    format: Literal["alpaca", "gemma", "sharegpt"] = Query("sharegpt"),
):
    async with _client() as c:
        r = await c.get(f"/api/provider/text/{dataset_id}/export", params={"format": format})
    _check(r)
    return r.json()


@router.post("/text/{dataset_id}/import", summary="Pull a text dataset from DataSupportTool and register it locally")
async def import_text_dataset(
    dataset_id: int,
    format: Literal["alpaca", "sharegpt"] = Query("sharegpt"),
    db: Session = Depends(get_db),
):
    async with _client() as c:
        r = await c.get(f"/api/provider/text/{dataset_id}/export", params={"format": format})
    _check(r)
    payload = r.json()

    records: list = payload.get("records", [])
    if not records:
        raise HTTPException(
            status_code=422,
            detail="No verified records available in this dataset yet — nothing to import",
        )

    dataset_name: str = payload.get("dataset_name", f"provider_text_{dataset_id}")
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in dataset_name)
    filename = f"provider_{safe}_{dataset_id}_{format}.jsonl"
    filepath = os.path.join(DATASETS_DIR, filename)

    def _write():
        with open(filepath, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.to_thread(_write)

    existing = db.query(Dataset).filter(Dataset.path == filepath).first()
    if existing:
        existing.num_samples = len(records)
        existing.description = (
            f"Imported from DataSupportTool '{dataset_name}' (id={dataset_id}, format={format})"
        )
        db.commit()
        return {"id": existing.id, "name": existing.name, "path": filepath, "record_count": len(records), "updated": True}

    ds = Dataset(
        name=f"{dataset_name} [{format}]",
        path=filepath,
        format=format,
        num_samples=len(records),
        description=f"Imported from DataSupportTool '{dataset_name}' (id={dataset_id}, format={format})",
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return {"id": ds.id, "name": ds.name, "path": filepath, "record_count": len(records), "updated": False}


# ── ASR dataset ────────────────────────────────────────────────────────────────

@router.get("/asr/{dataset_id}/export", summary="Proxy an ASR export from DataSupportTool (no import)")
async def proxy_asr_export(dataset_id: int):
    async with _client() as c:
        r = await c.get(f"/api/provider/asr/{dataset_id}/export")
    _check(r)
    return r.json()


@router.get("/asr/files/{file_id}/audio", summary="Proxy audio bytes for a specific ASR file")
async def proxy_audio_file(file_id: int):
    async with _client() as c:
        r = await c.get(f"/api/provider/asr/files/{file_id}/audio")
    _check(r)
    content_type = r.headers.get("content-type", "audio/wav")
    return Response(content=r.content, media_type=content_type)


@router.post("/asr/{dataset_id}/import", summary="Pull an ASR dataset from DataSupportTool and register it locally")
async def import_asr_dataset(dataset_id: int, db: Session = Depends(get_db)):
    async with _client() as c:
        r = await c.get(f"/api/provider/asr/{dataset_id}/export")
    _check(r)
    payload = r.json()

    records: list = payload.get("records", [])
    if not records:
        raise HTTPException(
            status_code=422,
            detail="No completed ASR records available yet — nothing to import",
        )

    dataset_name: str = payload.get("dataset_name", f"provider_asr_{dataset_id}")
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in dataset_name)
    filename = f"provider_asr_{safe}_{dataset_id}.jsonl"
    filepath = os.path.join(DATASETS_DIR, filename)

    def _write():
        with open(filepath, "w", encoding="utf-8") as f:
            for rec in records:
                row = {
                    "audio_path": rec.get("audio_url", ""),
                    "text": rec.get("transcript", ""),
                    "duration": rec.get("duration"),
                    "provider_file_id": rec.get("id"),
                    "filename": rec.get("filename", ""),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    await asyncio.to_thread(_write)

    existing = db.query(Dataset).filter(Dataset.path == filepath).first()
    if existing:
        existing.num_samples = len(records)
        db.commit()
        return {"id": existing.id, "name": existing.name, "path": filepath, "record_count": len(records), "updated": True}

    ds = Dataset(
        name=f"{dataset_name} [ASR]",
        path=filepath,
        format="asr_csv",
        num_samples=len(records),
        description=f"Imported ASR dataset from DataSupportTool (id={dataset_id})",
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return {"id": ds.id, "name": ds.name, "path": filepath, "record_count": len(records), "updated": False}


# ── BR Pipeline (read-only proxy — not for training input) ────────────────────

@router.get("/br-pipeline/{pipeline_run_id}/export", summary="Proxy a BR pipeline export (DataSupportTool's own Stage-5 outputs)")
async def proxy_br_pipeline_export(pipeline_run_id: int):
    async with _client() as c:
        r = await c.get(f"/api/provider/br-pipeline/{pipeline_run_id}/export")
    _check(r)
    return r.json()
