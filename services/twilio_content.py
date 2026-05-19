from __future__ import annotations

import json
import os

import requests

from observability.logger import log_step


TWILIO_MESSAGES_TIMEOUT_SECONDS = 5
MAX_LOGGED_RESPONSE_BODY_CHARS = 300


def _mask_phone_number(value: str | None) -> str:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    if len(digits) <= 4:
        return digits
    return f"...{digits[-4:]}"


def send_whatsapp_content_template(
    *,
    to_number: str,
    content_sid: str,
    from_number: str | None = None,
    content_variables: dict | None = None,
) -> bool:
    account_sid = (os.getenv("TWILIO_ACCOUNT_SID") or "").strip()
    auth_token = (os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    twilio_from = (
        (os.getenv("TWILIO_PHONE") or "").strip()
        or (from_number or "").strip()
    )

    if not account_sid or not auth_token:
        log_step("TWILIO_CONTENT", "Credenciales Twilio no configuradas")
        return False

    if not twilio_from:
        log_step("TWILIO_CONTENT", "Numero emisor de Twilio no configurado")
        return False

    if not to_number:
        log_step("TWILIO_CONTENT", "Numero destino ausente para template")
        return False

    if not content_sid:
        log_step("TWILIO_CONTENT", "ContentSid ausente para template")
        return False

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    payload = {
        "To": to_number,
        "From": twilio_from,
        "ContentSid": content_sid,
        "ContentVariables": json.dumps(content_variables or {}, ensure_ascii=False),
    }

    try:
        response = requests.post(
            url,
            data=payload,
            auth=(account_sid, auth_token),
            timeout=TWILIO_MESSAGES_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        log_step(
            "TWILIO_CONTENT",
            "Error enviando template CSAT",
            {
                "error": str(exc),
                "to": _mask_phone_number(to_number),
                "from": _mask_phone_number(twilio_from),
                "content_sid": content_sid,
            },
        )
        return False

    if 200 <= response.status_code < 300:
        log_step(
            "TWILIO_CONTENT",
            "Template CSAT enviado",
            {
                "to": _mask_phone_number(to_number),
                "from": _mask_phone_number(twilio_from),
                "content_sid": content_sid,
            },
        )
        return True

    log_step(
        "TWILIO_CONTENT",
        "Twilio rechazo template CSAT",
        {
            "status": response.status_code,
            "to": _mask_phone_number(to_number),
            "from": _mask_phone_number(twilio_from),
            "content_sid": content_sid,
            "body": response.text[:MAX_LOGGED_RESPONSE_BODY_CHARS],
        },
    )
    return False
