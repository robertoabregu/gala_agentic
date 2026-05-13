from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from langchain_openai import ChatOpenAI

from agents.state import AgentState
from core.privacy import (
    extract_identification,
    mask_identification,
    mask_sensitive_text,
)
from observability.logger import log_step
from tools.bcra_credit_status import query_bcra_credit_status


BCRA_INTERPRETIVE_TERMS = (
    "malo",
    "mala",
    "bueno",
    "buena",
    "grave",
    "normal",
    "significa",
    "riesgo",
    "afecta",
    "afecta mi credito",
    "credito",
    "deuda",
    "deudas",
    "complicado",
    "complicada",
    "situacion",
)

BCRA_BASE_SYSTEM_PROMPT = """
Sos Gala, asistente virtual de Banco Galicia.
Responde en espanol argentino, con tono claro, profesional y simple.
No inventes informacion.
No agregues datos que no esten en la informacion provista.
No des asesoramiento financiero personalizado.
No afirmes aprobacion o rechazo de productos bancarios.
No pidas claves, PIN, tokens, contrasenas, CVV ni datos sensibles.
No menciones endpoints, JSON, errores internos ni detalles tecnicos.
No cierres con preguntas innecesarias.
No uses expresiones especulativas como "creo que", "probablemente", "podria ser",
"en general", "normalmente" o "seguramente".
"""

BCRA_RESULT_SYSTEM_PROMPT = (
    BCRA_BASE_SYSTEM_PROMPT
    + """
Escenario: respuesta de consulta a la Central de Deudores del BCRA.
La informacion proviene de la Central de Deudores del BCRA y debes decirlo.
Usa solo los datos estructurados provistos.
Si hay datos:
- menciona el periodo mas reciente;
- menciona la denominacion si esta disponible;
- lista hasta 5 entidades;
- para cada entidad explica situacion, monto informado si existe y dias de atraso si existe;
- explica de forma simple que la situacion va de 1 a 5, donde 1 es normal y valores mas altos implican mayor nivel de riesgo informado.
Si no hay datos, di que no se encontraron datos para esa identificacion en la Central de Deudores del BCRA.
Si hay error tecnico, di que no se pudo consultar en este momento y sugiere intentar mas tarde.
No nombres datos que no figuren en la informacion provista.
"""
)

BCRA_FOLLOWUP_SYSTEM_PROMPT = (
    BCRA_BASE_SYSTEM_PROMPT
    + """
Escenario: follow-up interpretativo sobre una consulta previa de la Central de Deudores del BCRA.
Responde usando solo la pregunta actual y la memoria provista.
La respuesta debe ser general y prudente, sin asesoramiento personalizado.
Aclara que la informacion original proviene de la Central de Deudores del BCRA.
Si ayuda a responder, puedes explicar de forma simple que la situacion va de 1 a 5,
donde 1 es normal y valores mas altos implican mayor nivel de riesgo informado.
Si la pregunta apunta a impacto en el credito, responde de manera prudente:
la informacion del BCRA es una referencia y no implica por si sola aprobacion o rechazo.
Si la memoria no alcanza para responder, pide el CUIT o CUIL de 11 digitos en una sola oracion breve.
"""
)

BCRA_MISSING_ID_SYSTEM_PROMPT = (
    BCRA_BASE_SYSTEM_PROMPT
    + """
Escenario: falta el dato para consultar la Central de Deudores del BCRA.
Pide el CUIT o CUIL de 11 digitos en una sola oracion, breve y clara.
No agregues explicaciones innecesarias.
"""
)

BCRA_ERROR_SYSTEM_PROMPT = (
    BCRA_BASE_SYSTEM_PROMPT
    + """
Escenario: error tecnico al consultar la Central de Deudores del BCRA.
Di que no se pudo consultar en este momento y sugiere intentar mas tarde.
Responde en una o dos oraciones.
"""
)

BCRA_NO_DATA_SYSTEM_PROMPT = (
    BCRA_BASE_SYSTEM_PROMPT
    + """
Escenario: no se encontraron datos en la Central de Deudores del BCRA.
Di eso de forma breve y clara.
"""
)

MISSING_IDENTIFICATION_FALLBACK = (
    "Para consultar tu situacion crediticia necesito tu CUIT o CUIL de 11 digitos."
)
TECHNICAL_ERROR_FALLBACK = (
    "No pude consultar la Central de Deudores del BCRA en este momento. Intenta de nuevo mas tarde."
)
NO_DATA_FALLBACK = (
    "No se encontraron datos para esa identificacion en la Central de Deudores del BCRA."
)
FOLLOWUP_FALLBACK = (
    "La informacion de la Central de Deudores del BCRA se interpreta por niveles: "
    "la situacion 1 es normal y los valores mas altos indican mayor nivel de riesgo informado. "
    "Eso no implica por si solo aprobacion o rechazo."
)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    lowered = without_accents.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _llm_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(content).strip()


def _invoke_bcra_llm(
    llm: ChatOpenAI,
    *,
    system_prompt: str,
    payload: dict[str, Any],
    fallback_answer: str,
) -> str:
    try:
        response = llm.invoke(
            [
                ("system", system_prompt),
                ("user", json.dumps(payload, ensure_ascii=False)),
            ]
        )
        answer = _llm_text(response.content)
        return answer or fallback_answer
    except Exception as exc:
        log_step(
            "BCRA_AGENT",
            "Error generando respuesta con LLM",
            {"error": str(exc)},
        )
        return fallback_answer


def _extract_error_messages(payload: dict[str, Any]) -> list[str]:
    raw_messages = payload.get("errorMessages") or []

    if isinstance(raw_messages, str):
        return [raw_messages]

    if not isinstance(raw_messages, list):
        return []

    return [str(message) for message in raw_messages if str(message).strip()]


def _format_period(period: str) -> str:
    if len(period) == 6 and period.isdigit():
        return f"{period[4:6]}/{period[:4]}"
    return period


def _format_amount(amount: Any) -> str | None:
    if amount is None:
        return None

    try:
        amount_value = float(amount)
    except (TypeError, ValueError):
        return None

    if amount_value.is_integer():
        normalized_amount = str(int(amount_value))
    else:
        normalized_amount = f"{amount_value:.2f}".rstrip("0").rstrip(".")

    return f"{normalized_amount} miles de pesos"


def _extract_entity_flags(entity: dict[str, Any]) -> list[str]:
    flag_mapping = {
        "refinanciaciones": "refinanciaciones",
        "recategorizacionOblig": "recategorizacion obligatoria",
        "situacionJuridica": "situacion juridica",
        "irrecDisposicionTecnica": "irrecuperable por disposicion tecnica",
        "enRevision": "informacion en revision",
        "procesoJud": "proceso judicial",
    }

    return [
        label
        for field, label in flag_mapping.items()
        if entity.get(field)
    ]


def _build_bcra_summary(tool_result: dict[str, Any]) -> dict[str, Any]:
    payload = tool_result.get("data") or {}
    if not isinstance(payload, dict):
        payload = {}

    results = payload.get("results") or {}

    if not isinstance(results, dict):
        results = {}

    periods = results.get("periodos") or []
    if not isinstance(periods, list):
        periods = []

    normalized_periods = [
        period
        for period in periods
        if isinstance(period, dict)
    ]

    latest_period = (
        max(
            normalized_periods,
            key=lambda item: str(item.get("periodo", "")),
        )
        if normalized_periods
        else {}
    )

    entities = latest_period.get("entidades") or []
    if not isinstance(entities, list):
        entities = []

    normalized_entities = [
        entity
        for entity in entities
        if isinstance(entity, dict)
    ]

    summarized_entities = []
    for entity in normalized_entities[:5]:
        summarized_entities.append(
            {
                "entidad": str(entity.get("entidad") or "Entidad sin identificar"),
                "situacion": entity.get("situacion"),
                "monto_informado": _format_amount(entity.get("monto")),
                "dias_atraso_pago": entity.get("diasAtrasoPago"),
                "banderas": _extract_entity_flags(entity),
            }
        )

    latest_period_raw = str(latest_period.get("periodo", "")).strip()

    return {
        "status": tool_result.get("status"),
        "ok": bool(tool_result.get("ok")),
        "denominacion": results.get("denominacion"),
        "periodo_mas_reciente": latest_period_raw,
        "periodo_mas_reciente_formateado": _format_period(latest_period_raw),
        "total_entidades_periodo": len(normalized_entities),
        "entidades": summarized_entities,
        "entidades_adicionales": max(0, len(normalized_entities) - len(summarized_entities)),
        "error_messages": _extract_error_messages(payload),
    }


def _is_no_data_result(tool_result: dict[str, Any], summary: dict[str, Any]) -> bool:
    if not tool_result.get("ok"):
        return False

    if tool_result.get("status") == 404:
        return True

    normalized_errors = [
        message.lower()
        for message in summary.get("error_messages", [])
    ]

    if any(
        "no se encontro datos" in message
        or "no se encontraron datos" in message
        or "no se encontr" in message
        for message in normalized_errors
    ):
        return True

    return not summary.get("periodo_mas_reciente") or not summary.get("entidades")


def _is_bcra_memory_followup(state: AgentState, memory: dict[str, Any]) -> bool:
    if not state.get("is_followup"):
        return False

    last_route = str(memory.get("last_route", "")).strip()
    last_topic = str(memory.get("last_topic", "")).strip()
    last_assistant_answer = str(memory.get("last_assistant_answer", "")).strip()

    if not last_assistant_answer:
        return False

    if last_route != "bcra_credit_status" and last_topic != "situacion_crediticia_bcra":
        return False

    normalized_question = _normalize_text(
        state.get("standalone_question") or state.get("question", "")
    )

    return any(term in normalized_question for term in BCRA_INTERPRETIVE_TERMS)


def _build_followup_memory_payload(state: AgentState, memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "pregunta_actual": mask_sensitive_text(
            state.get("standalone_question") or state.get("question", "")
        ),
        "consulta_previa": {
            "last_user_question": mask_sensitive_text(memory.get("last_user_question", "")),
            "last_assistant_answer": mask_sensitive_text(memory.get("last_assistant_answer", "")),
            "last_route": str(memory.get("last_route", "")),
            "last_topic": str(memory.get("last_topic", "")),
        },
    }


def _build_missing_identification_payload(state: AgentState, memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "pregunta_actual": mask_sensitive_text(
            state.get("standalone_question") or state.get("question", "")
        ),
        "is_followup": bool(state.get("is_followup")),
        "last_route": str(memory.get("last_route", "")),
        "last_topic": str(memory.get("last_topic", "")),
    }


def bcra_agent_node(
    state: AgentState,
    llm: ChatOpenAI,
) -> AgentState:
    question = (state.get("standalone_question") or state.get("question") or "").strip()
    memory = state.get("memory") or {}
    identification = extract_identification(question)

    if not identification:
        if _is_bcra_memory_followup(state, memory):
            log_step("BCRA_AGENT", "Follow-up BCRA resuelto con memoria y LLM")
            answer = _invoke_bcra_llm(
                llm,
                system_prompt=BCRA_FOLLOWUP_SYSTEM_PROMPT,
                payload=_build_followup_memory_payload(state, memory),
                fallback_answer=FOLLOWUP_FALLBACK,
            )
            return {
                **state,
                "route": "bcra_credit_status",
                "tool_name": "bcra_credit_status",
                "tool_input": {},
                "tool_output": {
                    "memory_followup": True,
                    "last_route": str(memory.get("last_route", "")),
                    "last_topic": str(memory.get("last_topic", "")),
                    "last_user_question": mask_sensitive_text(memory.get("last_user_question", "")),
                    "last_assistant_answer": mask_sensitive_text(memory.get("last_assistant_answer", "")),
                },
                "needs_clarification": False,
                "missing_fields": [],
                "answer": answer,
                "error": None,
            }

        log_step("BCRA_AGENT", "Falta identificacion para consultar BCRA")
        answer = _invoke_bcra_llm(
            llm,
            system_prompt=BCRA_MISSING_ID_SYSTEM_PROMPT,
            payload=_build_missing_identification_payload(state, memory),
            fallback_answer=MISSING_IDENTIFICATION_FALLBACK,
        )
        return {
            **state,
            "route": "bcra_credit_status",
            "tool_name": "bcra_credit_status",
            "tool_input": {},
            "tool_output": {},
            "needs_clarification": True,
            "missing_fields": ["identificacion"],
            "answer": answer,
            "error": None,
        }

    masked_identification = mask_identification(identification)
    tool_result = query_bcra_credit_status(identification)

    base_state = {
        **state,
        "route": "bcra_credit_status",
        "tool_name": "bcra_credit_status",
        "tool_input": {"identificacion": masked_identification},
        "tool_output": tool_result,
        "needs_clarification": False,
        "missing_fields": [],
    }

    if not tool_result.get("ok"):
        log_step(
            "BCRA_AGENT",
            "Consulta BCRA con error tecnico",
            {
                "identificacion": masked_identification,
                "status": tool_result.get("status"),
                "error_type": tool_result.get("error_type"),
            },
        )
        answer = _invoke_bcra_llm(
            llm,
            system_prompt=BCRA_ERROR_SYSTEM_PROMPT,
            payload={
                "status": tool_result.get("status"),
                "error_type": tool_result.get("error_type"),
                "source": "Central de Deudores del BCRA",
            },
            fallback_answer=TECHNICAL_ERROR_FALLBACK,
        )
        return {
            **base_state,
            "answer": answer,
            "error": str(tool_result.get("error") or TECHNICAL_ERROR_FALLBACK),
        }

    summary = _build_bcra_summary(tool_result)

    if _is_no_data_result(tool_result, summary):
        log_step(
            "BCRA_AGENT",
            "BCRA sin datos para la identificacion consultada",
            {"identificacion": masked_identification},
        )
        answer = _invoke_bcra_llm(
            llm,
            system_prompt=BCRA_NO_DATA_SYSTEM_PROMPT,
            payload={
                "source": "Central de Deudores del BCRA",
                "status": tool_result.get("status"),
                "resultado": "sin_datos",
            },
            fallback_answer=NO_DATA_FALLBACK,
        )
        return {
            **base_state,
            "answer": answer,
            "error": None,
        }

    log_step(
        "BCRA_AGENT",
        "Generando respuesta BCRA con LLM",
        {
            "identificacion": masked_identification,
            "periodo": summary.get("periodo_mas_reciente_formateado", ""),
            "entidades": summary.get("total_entidades_periodo", 0),
        },
    )

    answer = _invoke_bcra_llm(
        llm,
        system_prompt=BCRA_RESULT_SYSTEM_PROMPT,
        payload={
            "source": "Central de Deudores del BCRA",
            "consulta": summary,
        },
        fallback_answer=TECHNICAL_ERROR_FALLBACK,
    )

    return {
        **base_state,
        "answer": answer,
        "error": None,
    }
