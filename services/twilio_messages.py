from __future__ import annotations

import os
from typing import Any

from observability.logger import log_step

try:
    from twilio.rest import Client
except ImportError:  # pragma: no cover - fallback solo para tests sin dependencias
    Client = Any  # type: ignore[misc,assignment]


def send_whatsapp_message(
    *,
    to_number: str,
    body: str,
    media_url: str | None = None,
    status_callback: str | None = None,
    from_number: str | None = None,
) -> str | None:
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    effective_from = (from_number or os.getenv("TWILIO_WHATSAPP_FROM", "")).strip()

    if not account_sid or not auth_token or not effective_from:
        log_step(
            "TWILIO_MESSAGES",
            "Credenciales o remitente de Twilio no configurados",
            {"has_from_number": bool(effective_from)},
        )
        return None

    client = Client(account_sid, auth_token)
    payload: dict[str, Any] = {
        "from_": effective_from,
        "to": to_number,
        "body": body,
    }

    effective_media_url = str(media_url or "").strip()
    if effective_media_url:
        payload["media_url"] = [effective_media_url]

    effective_status_callback = str(status_callback or "").strip()
    if effective_status_callback:
        payload["status_callback"] = effective_status_callback

    message = client.messages.create(**payload)
    message_sid = str(getattr(message, "sid", "") or "").strip() or None
    log_step(
        "TWILIO_MESSAGES",
        "Mensaje de WhatsApp enviado por API",
        {
            "to_suffix": str(to_number or "")[-4:],
            "has_media": bool(effective_media_url),
            "message_sid": message_sid or "",
        },
    )
    return message_sid
