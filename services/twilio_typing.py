from __future__ import annotations

import os

import requests

from observability.logger import log_step


TWILIO_TYPING_URL = "https://messaging.twilio.com/v2/Indicators/Typing.json"
TWILIO_TYPING_TIMEOUT_SECONDS = 3
MAX_LOGGED_RESPONSE_BODY_CHARS = 300


def send_whatsapp_typing_indicator(message_sid: str | None) -> bool:
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if not account_sid or not auth_token:
        log_step("TWILIO_TYPING", "Credenciales Twilio no configuradas")
        return False

    if not message_sid:
        log_step("TWILIO_TYPING", "MessageSid ausente; no se envia typing")
        return False

    if not str(message_sid).startswith(("SM", "MM")):
        log_step(
            "TWILIO_TYPING",
            "MessageSid invalido para typing",
            {"message_sid_prefix": str(message_sid)[:2]},
        )
        return False

    try:
        response = requests.post(
            TWILIO_TYPING_URL,
            data={
                "messageId": message_sid,
                "channel": "whatsapp",
            },
            auth=(account_sid, auth_token),
            timeout=TWILIO_TYPING_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        log_step(
            "TWILIO_TYPING",
            "Error enviando typing indicator",
            {"error": str(exc)},
        )
        return False

    if 200 <= response.status_code < 300:
        log_step("TWILIO_TYPING", "Typing indicator enviado")
        return True

    log_step(
        "TWILIO_TYPING",
        "Twilio rechazo typing indicator",
        {
            "status": response.status_code,
            "body": response.text[:MAX_LOGGED_RESPONSE_BODY_CHARS],
        },
    )
    return False
