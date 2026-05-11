from typing import Any
import time

import requests

from agents.state import AgentState
from core.privacy import extract_identification, mask_identification
from observability.logger import log_step


BCRA_TIMEOUT_SECONDS = 15
BCRA_MAX_RETRIES = 3
BCRA_RETRY_DELAY_SECONDS = 1

BCRA_DEBT_URL = "https://api.bcra.gob.ar/CentralDeDeudores/v1.0/Deudas/{identificacion}"

BCRA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

MISSING_IDENTIFICATION_MESSAGE = (
    "Para consultar tu situación crediticia necesito que me indiques tu CUIT o CUIL de 11 dígitos."
)


def _safe_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload
    except ValueError:
        pass

    return {
        "status": response.status_code,
        "errorMessages": ["La API del BCRA devolvió una respuesta no válida."],
    }


def _get_bcra_debt(url: str) -> requests.Response:
    last_error: Exception | None = None

    for attempt in range(1, BCRA_MAX_RETRIES + 1):
        try:
            return requests.get(
                url,
                timeout=BCRA_TIMEOUT_SECONDS,
                headers=BCRA_HEADERS,
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
            time.sleep(BCRA_RETRY_DELAY_SECONDS)

    if last_error:
        raise last_error

    raise requests.RequestException("No se pudo consultar BCRA.")


def bcra_tool_node(state: AgentState) -> AgentState:
    question = state.get("standalone_question") or state.get("question", "")
    memory = state.get("memory") or {}
    identification = extract_identification(question)

    if not identification:
        last_assistant_answer = str(memory.get("last_assistant_answer", ""))

        if (
            state.get("is_followup")
            and memory.get("last_route") == "bcra_credit_status"
            and last_assistant_answer
            and (
                "central de deudores del bcra" in last_assistant_answer.lower()
                or "situación " in last_assistant_answer.lower()
                or "situacion " in last_assistant_answer.lower()
            )
        ):
            log_step("BCRA_TOOL", "Follow-up BCRA resuelto con memoria local")
            return {
                **state,
                "tool_name": "bcra_credit_status",
                "tool_input": {},
                "tool_output": {
                    "status": 200,
                    "memory_followup": True,
                    "last_user_question": memory.get("last_user_question", ""),
                    "last_assistant_answer": last_assistant_answer,
                },
                "needs_clarification": False,
                "missing_fields": [],
                "error": None,
            }

        log_step("BCRA_TOOL", "Falta identificación para consultar BCRA")
        return {
            **state,
            "tool_name": "bcra_credit_status",
            "tool_input": {},
            "tool_output": {},
            "needs_clarification": True,
            "missing_fields": ["identificacion"],
            "answer": MISSING_IDENTIFICATION_MESSAGE,
            "final_answer": MISSING_IDENTIFICATION_MESSAGE,
            "error": None,
        }

    masked_identification = mask_identification(identification)
    url = BCRA_DEBT_URL.format(identificacion=identification)

    log_step(
        "BCRA_TOOL",
        "Consultando Central de Deudores del BCRA",
        {"identificacion": masked_identification},
    )

    try:
        response = _get_bcra_debt(url)
    except requests.Timeout:
        log_step("BCRA_TOOL", "Timeout definitivo consultando BCRA")
        return {
            **state,
            "tool_name": "bcra_credit_status",
            "tool_input": {"identificacion": masked_identification},
            "tool_output": {},
            "needs_clarification": False,
            "missing_fields": [],
            "error": "No pude consultar la Central de Deudores del BCRA por timeout.",
        }
    except requests.RequestException as exc:
        log_step("BCRA_TOOL", "Error definitivo de red consultando BCRA", {"error": str(exc)})
        return {
            **state,
            "tool_name": "bcra_credit_status",
            "tool_input": {"identificacion": masked_identification},
            "tool_output": {},
            "needs_clarification": False,
            "missing_fields": [],
            "error": "No pude consultar la Central de Deudores del BCRA en este momento.",
        }

    payload = _safe_json(response)

    log_step(
        "BCRA_TOOL",
        "Respuesta recibida desde BCRA",
        {
            "identificacion": masked_identification,
            "status": response.status_code,
        },
    )

    error = None

    if response.status_code >= 500:
        error = "No pude consultar la Central de Deudores del BCRA en este momento."
    elif response.status_code >= 400:
        error = "La Central de Deudores del BCRA no pudo procesar la consulta."

    return {
        **state,
        "tool_name": "bcra_credit_status",
        "tool_input": {"identificacion": masked_identification},
        "tool_output": payload,
        "needs_clarification": False,
        "missing_fields": [],
        "error": error,
    }