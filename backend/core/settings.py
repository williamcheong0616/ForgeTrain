"""settings.py — small DB-backed key/value settings store.

DB values take precedence over env vars, so settings configured from the
UI (e.g. the Settings page) override whatever was set at deploy time.
"""
from __future__ import annotations
import os

from sqlalchemy.orm import Session

from backend.db.models import Setting

PROVIDER_BASE_URL_KEY = "provider_base_url"
PROVIDER_API_KEY_KEY = "provider_api_key"


def get_setting(db: Session, key: str, default: str | None = None) -> str | None:
    row = db.query(Setting).filter(Setting.key == key).first()
    if row and row.value:
        return row.value
    return default


def set_setting(db: Session, key: str, value: str) -> None:
    row = db.query(Setting).filter(Setting.key == key).first()
    if row:
        row.value = value
    else:
        db.add(Setting(key=key, value=value))
    db.commit()


def get_provider_base_url(db: Session) -> str:
    return get_setting(db, PROVIDER_BASE_URL_KEY, os.getenv("PROVIDER_BASE_URL", "")).rstrip("/")


def get_provider_api_key(db: Session) -> str:
    return get_setting(db, PROVIDER_API_KEY_KEY, os.getenv("PROVIDER_API_KEY", ""))
