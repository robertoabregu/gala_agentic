from __future__ import annotations

import time
from typing import Any

import requests

from core.privacy import mask_identification, normalize_identification
from observability.logger import log_step


BCRA_TIMEOUT_SECONDS = 10
BCRA_MAX_RETRIES = 3
BCRA_RETRY_DELAY_SECONDS = 1

BCRA_DEBT_URL = "https://api.bcra.gob.ar/CentralDeDeudores/v1.0/Deudas/{identificacion}"

BCRA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


def _safe_json(response: requests.Response) -> tuple[dict[str, Any], bool]:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload, True
    except ValueError:
        pass

    return {
        "status": response.status_code,
        "errorMessages": ["La API del BCRA devolvio una respuesta no valida."],
    }, False


def _request_bcra_credit_status(identificacion: str) -> requests.Response:
    url = BCRA_DEBT_URL.format(identificacion=identificacion)
    last_error: Exception | None = None
    last_response: requests.Response | None = None

    retryable_status_codes = {500, 502, 503, 504}

    for attempt in range(1, BCRA_MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                timeout=BCRA_TIMEOUT_SECONDS,
                headers=BCRA_HEADERS,
            )

            last_response = response

            if response.status_code not in retryable_status_codes:
                return response

            log_step(
                "BCRA_TOOL",
                f"BCRA devolvio status retryable intento {attempt}/{BCRA_MAX_RETRIES}",
                {"status": response.status_code},
            )

        except requests.Timeout as exc:
            last_error = exc
            log_step(
                "BCRA_TOOL",
                f"Timeout consultando BCRA intento {attempt}/{BCRA_MAX_RETRIES}",
            )

        except requests.RequestException as exc:
            last_error = exc
            log_step(
                "BCRA_TOOL",
                f"Error de red consultando BCRA intento {attempt}/{BCRA_MAX_RETRIES}",
                {"error": str(exc)},
            )

        if attempt < BCRA_MAX_RETRIES:
            time.sleep(BCRA_RETRY_DELAY_SECONDS * attempt)

    if last_response is not None:
        return last_response

    if last_error:
        raise last_error

    raise requests.RequestException("No se pudo consultar BCRA.")


def _extract_error_messages(payload: dict[str, Any]) -> list[str]:
    raw_messages = payload.get("errorMessages") or []

    if isinstance(raw_messages, str):
        return [raw_messages]

    if not isinstance(raw_messages, list):
        return []

    return [str(message) for message in raw_messages if str(message).strip()]


def _is_no_data_response(status: int | None, payload: dict[str, Any]) -> bool:
    if status == 404:
        return True

    error_messages = _extract_error_messages(payload)
    normalized_messages = [message.lower() for message in error_messages]

    return any(
        "no se encontro datos" in message
        or "no se encontraron datos" in message
        or "no se encontr" in message
        for message in normalized_messages
    )


def query_bcra_credit_status(identificacion: str) -> dict[str, Any]:
    normalized_identification = normalize_identification(identificacion)
    masked_identification = mask_identification(normalized_identification)

    if len(normalized_identification) != 11:
        log_step(
            "BCRA_TOOL",
            "Identificacion invalida para consulta BCRA",
            {"identificacion": masked_identification},
        )
        return {
            "ok": False,
            "status": None,
            "data": {},
            "error": "La identificacion para consultar BCRA es invalida.",
            "error_type": "validation_error",
        }

    log_step(
        "BCRA_TOOL",
        "Consultando Central de Deudores del BCRA",
        {"identificacion": masked_identification},
    )

    try:
        response = _request_bcra_credit_status(normalized_identification)
    except requests.Timeout:
        log_step(
            "BCRA_TOOL",
            "Timeout definitivo consultando BCRA",
            {"identificacion": masked_identification},
        )
        return {
            "ok": False,
            "status": None,
            "data": {},
            "error": "No se pudo consultar la Central de Deudores del BCRA por timeout.",
            "error_type": "timeout",
        }
    except requests.RequestException as exc:
        log_step(
            "BCRA_TOOL",
            "Error definitivo de red consultando BCRA",
            {
                "identificacion": masked_identification,
                "error": str(exc),
            },
        )
        return {
            "ok": False,
            "status": None,
            "data": {},
            "error": "No se pudo consultar la Central de Deudores del BCRA en este momento.",
            "error_type": "network_error",
        }

    payload, is_valid_json = _safe_json(response)
    status = response.status_code

    log_step(
        "BCRA_TOOL",
        "Respuesta recibida desde BCRA",
        {
            "identificacion": masked_identification,
            "status": status,
        },
    )

    if not is_valid_json:
        return {
            "ok": False,
            "status": status,
            "data": payload,
            "error": "La Central de Deudores del BCRA devolvio una respuesta no valida.",
            "error_type": "invalid_response",
        }

    if _is_no_data_response(status, payload):
        return {
            "ok": True,
            "status": status,
            "data": payload,
            "error": None,
            "error_type": None,
        }

    if status >= 500:
        return {
            "ok": False,
            "status": status,
            "data": payload,
            "error": "No se pudo consultar la Central de Deudores del BCRA en este momento.",
            "error_type": "upstream_error",
        }

    if status >= 400:
        return {
            "ok": False,
            "status": status,
            "data": payload,
            "error": "La Central de Deudores del BCRA no pudo procesar la consulta.",
            "error_type": "client_error",
        }

    return {
        "ok": True,
        "status": status,
        "data": payload,
        "error": None,
        "error_type": None,
    }
