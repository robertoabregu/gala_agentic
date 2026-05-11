from typing import Any
import re

from agents.state import AgentState
from observability.logger import log_step


SITUATION_LABELS = {
    1: "normal",
    2: "riesgo bajo",
    3: "riesgo medio",
    4: "riesgo alto",
    5: "irrecuperable",
}

SITUATION_EXPLANATIONS = {
    1: "En principio no: la situacion 1 suele indicar cumplimiento normal o sin atrasos relevantes informados.",
    2: "No es la mejor calificacion, pero suele indicar un nivel de riesgo bajo o atrasos leves informados.",
    3: "Es una senal de riesgo medio y puede reflejar atrasos mas significativos o un deterioro en el comportamiento de pago.",
    4: "Es una situacion de riesgo alto y suele reflejar incumplimientos importantes.",
    5: "Es la categoria mas negativa y suele asociarse a deuda considerada irrecuperable.",
}


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


def _build_entity_line(entity: dict[str, Any]) -> str:
    entity_name = entity.get("entidad") or "Entidad sin identificar"
    situation = entity.get("situacion")
    situation_label = SITUATION_LABELS.get(situation, "sin clasificacion informada")
    days_late = entity.get("diasAtrasoPago")
    amount = _format_amount(entity.get("monto"))

    details = [
        f"*situacion {situation}* ({situation_label})"
        if situation
        else situation_label
    ]

    if amount:
        details.append(f"*monto informado:* {amount}")

    if days_late is not None:
        details.append(f"*{days_late} dias de atraso*")

    flags = []

    if entity.get("refinanciaciones"):
        flags.append("refinanciaciones")

    if entity.get("recategorizacionOblig"):
        flags.append("recategorizacion obligatoria")

    if entity.get("situacionJuridica"):
        flags.append("situacion juridica")

    if entity.get("irrecDisposicionTecnica"):
        flags.append("irrecuperable por disposicion tecnica")

    if entity.get("enRevision"):
        flags.append("informacion en revision")

    if entity.get("procesoJud"):
        flags.append("proceso judicial")

    if flags:
        details.append(
            "*ademas figura:* " + ", ".join(flags)
        )

    return f"- *{entity_name}*: " + ", ".join(details) + "."


def _extract_situation_from_text(text: str) -> int | None:
    match = re.search(r"situacion\s+([1-5])", text or "", flags=re.IGNORECASE)
    if not match:
        return None

    try:
        return int(match.group(1))
    except ValueError:
        return None


def _build_memory_followup_answer(state: AgentState, tool_output: dict[str, Any]) -> str:
    standalone_question = (state.get("standalone_question") or state.get("question") or "").lower()
    last_assistant_answer = tool_output.get("last_assistant_answer", "")
    situation = _extract_situation_from_text(last_assistant_answer)

    if situation and any(
        pattern in standalone_question
        for pattern in ["malo", "mala", "grave", "bueno", "buena", "normal", "significa"]
    ):
        explanation = SITUATION_EXPLANATIONS.get(
            situation,
            "La clasificacion del BCRA se interpreta por niveles de riesgo y cuanto mas alto es el numero, mas delicada suele ser la situacion.",
        )
        return (
            f"{explanation} "
            "La informacion se interpreta segun la Central de Deudores del BCRA."
        )

    if last_assistant_answer:
        return (
            "Tomando como referencia la consulta anterior en la Central de Deudores del BCRA: "
            f"{last_assistant_answer}\n"
            "Si queres, tambien puedo ayudarte a interpretar esa situacion en terminos generales."
        )

    return (
        "No tengo el detalle completo de la consulta anterior, pero en la Central de Deudores del BCRA "
        "las situaciones van de 1 a 5 y cuanto mas alto es el numero, mas delicada suele ser la clasificacion."
    )


def bcra_answer_node(state: AgentState) -> AgentState:
    if state.get("needs_clarification"):
        answer = state.get("final_answer") or state.get("answer") or ""
        log_step("BCRA_ANSWER", "Se solicita aclaracion al usuario")
        return {
            **state,
            "answer": answer,
        }

    if state.get("error"):
        answer = (
            "No pude consultar la Central de Deudores del BCRA en este momento. "
            "Proba de nuevo mas tarde."
        )
        log_step("BCRA_ANSWER", "No se pudo responder por error de la herramienta")
        return {
            **state,
            "answer": answer,
        }

    tool_output = state.get("tool_output") or {}

    if tool_output.get("memory_followup"):
        answer = _build_memory_followup_answer(state, tool_output)
        log_step("BCRA_ANSWER", "Follow-up BCRA respondido con memoria local")
        return {
            **state,
            "answer": answer,
        }

    status = tool_output.get("status")
    error_messages = tool_output.get("errorMessages") or []

    normalized_errors = [str(message).lower() for message in error_messages]

    if status == 404 or any(
        "no se encontro datos" in message or "no se encontr" in message
        for message in normalized_errors
    ):
        answer = (
            "No encontre datos para esa identificacion en la Central de Deudores del BCRA."
        )
        log_step("BCRA_ANSWER", "BCRA sin datos para la identificacion consultada")
        return {
            **state,
            "answer": answer,
        }

    if status and status != 200:
        answer = (
            "No pude consultar la Central de Deudores del BCRA en este momento. "
            "Proba de nuevo mas tarde."
        )
        log_step("BCRA_ANSWER", "BCRA devolvio un estado no exitoso", {"status": status})
        return {
            **state,
            "answer": answer,
        }

    results = tool_output.get("results") or {}
    periods = results.get("periodos") or []

    if not periods:
        answer = (
            "No encontre datos para esa identificacion en la Central de Deudores del BCRA."
        )
        log_step("BCRA_ANSWER", "BCRA respondio sin periodos")
        return {
            **state,
            "answer": answer,
        }

    latest_period = max(
        periods,
        key=lambda period_data: str(period_data.get("periodo", "")),
    )
    formatted_period = _format_period(str(latest_period.get("periodo", "")))
    entities = latest_period.get("entidades") or []

    if not entities:
        answer = (
            "No encontre financiaciones informadas para esa identificacion en la Central de Deudores del BCRA."
        )
        log_step("BCRA_ANSWER", "BCRA respondio sin entidades")
        return {
            **state,
            "answer": answer,
        }

    denomination = results.get("denominacion")
    intro = "Segun la Central de Deudores del BCRA"
    if denomination:
        intro += f", la informacion disponible para {denomination}"
    intro += f" corresponde al periodo {formatted_period}:"

    entity_lines = [_build_entity_line(entity) for entity in entities[:5]]
    if len(entities) > 5:
        entity_lines.append(
            f"- Hay {len(entities) - 5} entidades adicionales informadas en ese periodo."
        )

    answer = "\n".join(
        [
            intro,
            *entity_lines,
            "La informacion proviene de la Central de Deudores del BCRA.",
        ]
    )

    log_step(
        "BCRA_ANSWER",
        "Respuesta BCRA resumida",
        {"periodo": formatted_period, "entidades": len(entities)},
    )

    return {
        **state,
        "answer": answer,
    }
