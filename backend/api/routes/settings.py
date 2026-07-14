"""settings.py — UI-editable app settings (currently: DataSupportTool provider connection)."""
from __future__ import annotations
import secrets

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.core.settings import (
    PROVIDER_API_KEY_KEY,
    PROVIDER_BASE_URL_KEY,
    get_provider_api_key,
    get_provider_base_url,
    set_setting,
)

router = APIRouter(prefix="/api/settings", tags=["settings"])


class ProviderSettings(BaseModel):
    base_url: str
    api_key: str
    configured: bool


class ProviderSettingsUpdate(BaseModel):
    base_url: str | None = None
    api_key: str | None = None


def _provider_settings(db: Session) -> ProviderSettings:
    base_url = get_provider_base_url(db)
    api_key = get_provider_api_key(db)
    return ProviderSettings(base_url=base_url, api_key=api_key, configured=bool(base_url and api_key))


@router.get("/provider", response_model=ProviderSettings, summary="Get DataSupportTool connection settings")
def get_provider_settings(db: Session = Depends(get_db)):
    return _provider_settings(db)


@router.put("/provider", response_model=ProviderSettings, summary="Update DataSupportTool endpoint and/or API key")
def update_provider_settings(body: ProviderSettingsUpdate, db: Session = Depends(get_db)):
    if body.base_url is not None:
        set_setting(db, PROVIDER_BASE_URL_KEY, body.base_url.strip().rstrip("/"))
    if body.api_key is not None:
        set_setting(db, PROVIDER_API_KEY_KEY, body.api_key.strip())
    return _provider_settings(db)


@router.post(
    "/provider/generate-key",
    response_model=ProviderSettings,
    summary="Generate a new shared secret for the DataSupportTool provider API key",
)
def generate_provider_key(db: Session = Depends(get_db)):
    key = secrets.token_hex(32)
    set_setting(db, PROVIDER_API_KEY_KEY, key)
    return _provider_settings(db)
