from __future__ import annotations

import json
import os
from typing import Any

from twilio.rest import Client

from observability.logger import log_step


DEFAULT_WHATSAPP_CONTENT_VARIABLES = {"1": " "}


def _mask_phone_number(value: str | None) -> str:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    if len(digits) <= 4:
        return digits
    return f"...{digits[-4:]}"


def _sanitize_content_variables_dict(content_variables: dict[str, Any]) -> dict[str, str]:
    sanitized_variables: dict[str, str] = {}
    for key, value in content_variables.items():
        if value is None:
            continue

        text_value = str(value).strip()
        if not text_value:
            continue

        sanitized_variables[str(key)] = text_value

    return sanitized_variables


def _serialize_content_variables(content_variables: dict[str, Any] | str | None) -> str:
    if isinstance(content_variables, str):
        stripped_content_variables = content_variables.strip()
        if stripped_content_variables:
            return stripped_content_variables

        return json.dumps(DEFAULT_WHATSAPP_CONTENT_VARIABLES, ensure_ascii=False)

    if isinstance(content_variables, dict):
        sanitized_variables = _sanitize_content_variables_dict(content_variables)
        if sanitized_variables:
            return json.dumps(sanitized_variables, ensure_ascii=False)

    return json.dumps(DEFAULT_WHATSAPP_CONTENT_VARIABLES, ensure_ascii=False)


def _get_content_variable_keys(content_variables: dict[str, Any] | str | None) -> list[str]:
    if isinstance(content_variables, dict):
        return sorted(_sanitize_content_variables_dict(content_variables).keys())

    if isinstance(content_variables, str):
        try:
            parsed = json.loads(content_variables)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed, dict):
            return sorted(_sanitize_content_variables_dict(parsed).keys())

    return sorted(DEFAULT_WHATSAPP_CONTENT_VARIABLES.keys())


def send_whatsapp_content_template(
    *,
    to_number: str,
    content_sid: str,
    from_number: str | None = None,
    content_variables: dict[str, Any] | str | None = None,
) -> bool:
    account_sid = (os.getenv("TWILIO_ACCOUNT_SID") or "").strip()
    auth_token = (os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    twilio_from = (
        (os.getenv("TWILIO_PHONE") or "").strip()
        or (from_number or "").strip()
    )
    twilio_messaging_sid = (os.getenv("TWILIO_MESSAGING_SID") or "").strip()

    if not account_sid or not auth_token:
        log_step("TWILIO_CONTENT", "Credenciales Twilio no configuradas")
        return False

    if not twilio_messaging_sid and not twilio_from:
        log_step("TWILIO_CONTENT", "Numero emisor o Messaging Service de Twilio no configurado")
        return False

    if not to_number:
        log_step("TWILIO_CONTENT", "Numero destino ausente para template")
        return False

    if not content_sid:
        log_step("TWILIO_CONTENT", "ContentSid ausente para template")
        return False

    serialized_content_variables = _serialize_content_variables(content_variables)
    content_variable_keys = _get_content_variable_keys(content_variables)

    msg_kwargs: dict[str, Any] = {
        "to": to_number,
        "content_sid": content_sid,
        "content_variables": serialized_content_variables,
    }

    if twilio_messaging_sid:
        msg_kwargs["messaging_service_sid"] = twilio_messaging_sid
    else:
        msg_kwargs["from_"] = twilio_from

    try:
        client = Client(account_sid, auth_token)
        sent_message = client.messages.create(**msg_kwargs)
    except Exception as exc:
        log_step(
            "TWILIO_CONTENT",
            "Error enviando template CSAT",
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "to": _mask_phone_number(to_number),
                "from": _mask_phone_number(twilio_from),
                "messaging_service_sid": twilio_messaging_sid,
                "content_sid": content_sid,
                "content_variable_keys": content_variable_keys,
            },
        )
        return False

    log_step(
        "TWILIO_CONTENT",
        "Template CSAT enviado",
        {
            "to": _mask_phone_number(to_number),
            "from": _mask_phone_number(twilio_from),
            "messaging_service_sid": twilio_messaging_sid,
            "content_sid": content_sid,
            "content_variable_keys": content_variable_keys,
            "status": getattr(sent_message, "status", ""),
            "error_code": getattr(sent_message, "error_code", None),
            "error_message": getattr(sent_message, "error_message", None),
        },
    )
    return True
