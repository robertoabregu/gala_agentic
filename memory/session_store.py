from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
SESSION_STORE_PATH = BASE_DIR / "data" / "session_store" / "whatsapp_sessions.json"
DEFAULT_SESSION_TTL_SECONDS = 900
_SESSION_STORE_LOCK = threading.RLock()


def get_or_create_conversation_session(user_id: str) -> str:
    normalized_user_id = _normalize_user_id(user_id)
    now = _utcnow()
    ttl_seconds = _session_ttl_seconds()

    with _SESSION_STORE_LOCK:
        store = _load_store_unlocked()
        existing_record = _normalize_session_record(store.get(normalized_user_id))

        if _is_session_active(existing_record, now, ttl_seconds):
            active_session_id = str(existing_record["active_session_id"])
            existing_record["last_seen_at"] = _isoformat_utc(now)
            store[normalized_user_id] = existing_record
            _persist_store_safely(store)
            return active_session_id

        new_session_id = _build_conversation_session_id(normalized_user_id, now)
        store[normalized_user_id] = {
            "active_session_id": new_session_id,
            "created_at": _isoformat_utc(now),
            "last_seen_at": _isoformat_utc(now),
        }
        _persist_store_safely(store)
        return new_session_id


def _normalize_user_id(user_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", (user_id or "").strip())
    return cleaned or "whatsapp-anonymous"


def _session_ttl_seconds() -> int:
    raw_value = os.getenv("WHATSAPP_SESSION_TTL_SECONDS", str(DEFAULT_SESSION_TTL_SECONDS))
    try:
        return max(0, int(raw_value or str(DEFAULT_SESSION_TTL_SECONDS)))
    except (TypeError, ValueError):
        return DEFAULT_SESSION_TTL_SECONDS


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _build_conversation_session_id(user_id: str, now: datetime) -> str:
    timestamp = now.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{user_id}-{timestamp}-{secrets.token_hex(2)}"


def _load_store_unlocked() -> dict[str, Any]:
    if not SESSION_STORE_PATH.exists():
        return {}

    try:
        payload = json.loads(SESSION_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log_store_error(f"load failed: {type(exc).__name__}")
        return {}

    if isinstance(payload, dict):
        if isinstance(payload.get("users"), dict):
            return dict(payload["users"])
        if "users" in payload and not isinstance(payload.get("users"), dict):
            return {}
        return dict(payload)

    return {}


def _normalize_session_record(record: Any) -> dict[str, str] | None:
    if not isinstance(record, dict):
        return None

    active_session_id = str(record.get("active_session_id") or "").strip()
    created_at = _parse_utc_datetime(record.get("created_at"))
    last_seen_at = _parse_utc_datetime(record.get("last_seen_at"))

    if not active_session_id or created_at is None or last_seen_at is None:
        return None

    return {
        "active_session_id": active_session_id,
        "created_at": _isoformat_utc(created_at),
        "last_seen_at": _isoformat_utc(last_seen_at),
    }


def _parse_utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _is_session_active(
    record: dict[str, str] | None,
    now: datetime,
    ttl_seconds: int,
) -> bool:
    if record is None or ttl_seconds <= 0:
        return False

    last_seen_at = _parse_utc_datetime(record.get("last_seen_at"))
    if last_seen_at is None:
        return False

    return now <= last_seen_at + timedelta(seconds=ttl_seconds)


def _persist_store_safely(store: dict[str, Any]) -> None:
    try:
        _save_store_unlocked(store)
    except OSError as exc:
        _log_store_error(f"save failed: {type(exc).__name__}")


def _save_store_unlocked(store: dict[str, Any]) -> None:
    SESSION_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = SESSION_STORE_PATH.with_suffix(f".{secrets.token_hex(4)}.tmp")

    try:
        temp_path.write_text(
            json.dumps(store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(SESSION_STORE_PATH)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def _log_store_error(message: str) -> None:
    print(f"[session-store] {message}")
