from __future__ import annotations

import json
import os
from typing import Any

import requests

from observability.logger import log_step


TWILIO_MESSAGES_TIMEOUT_SECONDS = 5
MAX_LOGGED_RESPONSE_BODY_CHARS = 300
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

    serialized_content_variables = _serialize_content_variables(content_variables)
    content_variable_keys = _get_content_variable_keys(content_variables)

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    payload = {
        "To": to_number,
        "From": twilio_from,
        "ContentSid": content_sid,
        "ContentVariables": serialized_content_variables,
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
                "error_type": type(exc).__name__,
                "to": _mask_phone_number(to_number),
                "from": _mask_phone_number(twilio_from),
                "content_sid": content_sid,
                "content_variable_keys": content_variable_keys,
            },
        )
        return False

    if 200 <= response.status_code < 300:
        log_step(
            "TWILIO_CONTENT",
            "Template CSAT enviado",
            {
                "status": response.status_code,
                "to": _mask_phone_number(to_number),
                "from": _mask_phone_number(twilio_from),
                "content_sid": content_sid,
                "content_variable_keys": content_variable_keys,
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
            "content_variable_keys": content_variable_keys,
            "body": response.text[:MAX_LOGGED_RESPONSE_BODY_CHARS],
        },
    )
    return False
