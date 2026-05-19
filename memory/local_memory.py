import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
MEMORY_DIR = BASE_DIR / "data" / "memory"

DEFAULT_MEMORY = {
    "user_name": "",
    "pending_route": "",
    "missing_fields": [],
    "last_route": "",
    "last_user_question": "",
    "last_assistant_answer": "",
    "last_topic": "",
    "updated_at": "",
    "credit_card_statement": {},
    "csat_sent": False,
    "csat_sent_at": "",
    "csat_template_sid": "",
}


def _sanitize_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", (session_id or "").strip())
    return cleaned or "demo-local"


def _memory_path(session_id: str) -> Path:
    safe_session_id = _sanitize_session_id(session_id)
    return MEMORY_DIR / f"{safe_session_id}.json"


def _normalize_memory(memory: dict[str, Any] | None) -> dict[str, Any]:
    # IMPORTANTE:
    # Partimos de la memoria existente para no perder campos nuevos
    # como user_name u otros datos que agreguemos más adelante.
    normalized = dict(memory) if isinstance(memory, dict) else {}

    for key, default_value in DEFAULT_MEMORY.items():
        normalized.setdefault(key, default_value)

    normalized["user_name"] = (
        normalized["user_name"]
        if isinstance(normalized.get("user_name"), str)
        else ""
    )

    normalized["pending_route"] = (
        normalized["pending_route"]
        if isinstance(normalized.get("pending_route"), str)
        else ""
    )

    normalized["missing_fields"] = (
        [str(field) for field in normalized.get("missing_fields", [])]
        if isinstance(normalized.get("missing_fields"), list)
        else []
    )

    normalized["last_route"] = (
        normalized["last_route"]
        if isinstance(normalized.get("last_route"), str)
        else ""
    )

    normalized["last_user_question"] = (
        normalized["last_user_question"]
        if isinstance(normalized.get("last_user_question"), str)
        else ""
    )

    normalized["last_assistant_answer"] = (
        normalized["last_assistant_answer"]
        if isinstance(normalized.get("last_assistant_answer"), str)
        else ""
    )

    normalized["last_topic"] = (
        normalized["last_topic"]
        if isinstance(normalized.get("last_topic"), str)
        else ""
    )

    normalized["updated_at"] = (
        normalized["updated_at"]
        if isinstance(normalized.get("updated_at"), str)
        else ""
    )

    normalized["credit_card_statement"] = (
        normalized["credit_card_statement"]
        if isinstance(normalized.get("credit_card_statement"), dict)
        else {}
    )

    normalized["csat_sent"] = bool(normalized.get("csat_sent"))

    normalized["csat_sent_at"] = (
        normalized["csat_sent_at"]
        if isinstance(normalized.get("csat_sent_at"), str)
        else ""
    )

    normalized["csat_template_sid"] = (
        normalized["csat_template_sid"]
        if isinstance(normalized.get("csat_template_sid"), str)
        else ""
    )

    return normalized


def load_memory(session_id: str) -> dict[str, Any]:
    memory_path = _memory_path(session_id)

    if not memory_path.exists():
        return dict(DEFAULT_MEMORY)

    try:
        raw_memory = json.loads(memory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_MEMORY)

    return _normalize_memory(raw_memory)


def save_memory(session_id: str, memory: dict[str, Any]) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    memory_path = _memory_path(session_id)
    normalized_memory = _normalize_memory(memory)
    memory_path.write_text(
        json.dumps(normalized_memory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_pending(memory: dict[str, Any]) -> dict[str, Any]:
    updated_memory = _normalize_memory(memory)
    updated_memory["pending_route"] = ""
    updated_memory["missing_fields"] = []
    return updated_memory


def set_pending(
    memory: dict[str, Any],
    route: str,
    missing_fields: list[str],
) -> dict[str, Any]:
    updated_memory = _normalize_memory(memory)
    updated_memory["pending_route"] = route or ""
    updated_memory["missing_fields"] = [str(field) for field in missing_fields]
    return updated_memory


def mark_csat_sent(
    memory: dict[str, Any],
    *,
    template_sid: str,
    sent_at: str,
) -> dict[str, Any]:
    updated_memory = _normalize_memory(memory)
    updated_memory["csat_sent"] = True
    updated_memory["csat_sent_at"] = sent_at
    updated_memory["csat_template_sid"] = template_sid
    return updated_memory
