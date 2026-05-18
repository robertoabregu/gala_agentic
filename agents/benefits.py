from __future__ import annotations

import re
import unicodedata
from typing import Any

from agents.state import AgentState
from core.privacy import mask_sensitive_text
from observability.logger import log_step
from tools.benefits_api import (
    get_benefits_segment,
    infer_benefits_filters,
    list_benefit_categories,
    search_benefits,
)


CATEGORY_EMOJIS = {
    "Supermercados": "🛒",
    "Gastronomía": "🍽️",
    "Indumentaria": "👕",
    "Electrónica": "💻",
    "Hogar": "🏠",
    "Vehículos": "🚗",
    "Salud y Bienestar": "💚",
    "Viajes": "✈️",
    "Entretenimiento": "🎟️",
    "Librerías": "📚",
    "Shopping": "🛍️",
    "Mascotas": "🐾",
    "Juguetes": "🧸",
    "Transportes": "🚌",
    "Otros": "🎁",
}

REFERENCE_FOLLOWUP_PATTERNS = (
    "esos",
    "esas",
    "los mismos",
    "las mismas",
    "los de antes",
    "las de antes",
    "los anteriores",
    "las anteriores",
)


def benefits_node(state: AgentState) -> AgentState:
    question = (state.get("question") or "").strip()
    standalone_question = (state.get("standalone_question") or question).strip()

    try:
        filters = _resolve_benefits_filters(
            question=question,
            standalone_question=standalone_question,
            is_followup=bool(state.get("is_followup")),
        )

        if _should_show_categories_summary(filters):
            answer = _build_categories_answer()
            results: list[dict[str, Any]] = []
        else:
            results = search_benefits(
                category=filters["category"],
                query=filters["cleaned_query"],
                raw_query=question,
                only_eminent=filters["only_eminent"],
                exclude_eminent=filters["exclude_eminent"],
                only_qr=filters["only_qr"],
                only_nfc=filters["only_nfc"],
                today_only=filters["today_only"],
                every_day_only=filters["every_day_only"],
                installments=filters["installments"],
                has_installments=filters["has_installments"],
                interest_free=filters["interest_free"],
                search_terms=filters["search_terms"],
                limit=5,
            )

            if results:
                answer = _build_results_answer(results, filters)
            else:
                answer = _build_no_results_answer(filters)
    except Exception as exc:
        log_step(
            "BENEFITS",
            "Error consultando beneficios",
            {
                "question": mask_sensitive_text(question),
                "standalone_question": mask_sensitive_text(standalone_question),
                "error": str(exc),
            },
        )
        return {
            **state,
            "route": "benefits",
            "tool_name": "benefits_api",
            "tool_input": {
                "question": question,
                "standalone_question": standalone_question,
            },
            "tool_output": {
                "results_count": 0,
                "results": [],
            },
            "needs_clarification": False,
            "missing_fields": [],
            "pending_route": "",
            "answer": (
                "🔎 No pude consultar las promociones de Galicia en este momento. "
                "Si querés, probá de nuevo en un ratito."
            ),
            "topic": "beneficios",
            "error": str(exc),
        }

    log_step(
        "BENEFITS",
        "Filtros detectados",
        {
            "question": mask_sensitive_text(question),
            "standalone_question": mask_sensitive_text(standalone_question),
            "category": filters["category"],
            "raw_query": mask_sensitive_text(filters["raw_query"]),
            "cleaned_query": mask_sensitive_text(filters["cleaned_query"]),
            "search_terms": filters["search_terms"],
            "only_eminent": filters["only_eminent"],
            "exclude_eminent": filters["exclude_eminent"],
            "only_qr": filters["only_qr"],
            "only_nfc": filters["only_nfc"],
            "installments": filters["installments"],
            "has_installments": filters["has_installments"],
            "interest_free": filters["interest_free"],
            "results_count": len(results),
        },
    )

    return {
        **state,
        "route": "benefits",
        "tool_name": "benefits_api",
        "tool_input": {
            "question": question,
            "standalone_question": standalone_question,
            **filters,
        },
        "tool_output": {
            "results_count": len(results),
            "results": results,
        },
        "needs_clarification": False,
        "missing_fields": [],
        "pending_route": "",
        "answer": answer,
        "topic": "beneficios",
        "error": None,
    }


def _resolve_benefits_filters(
    *,
    question: str,
    standalone_question: str,
    is_followup: bool,
) -> dict[str, Any]:
    current_filters = infer_benefits_filters(question)
    standalone_filters = infer_benefits_filters(standalone_question)

    category = current_filters["category"]
    if not category and _can_use_standalone_context(question, current_filters, is_followup):
        category = standalone_filters["category"]

    exclude_eminent = bool(current_filters["exclude_eminent"])
    only_eminent = bool(current_filters["only_eminent"]) and not exclude_eminent
    installments = current_filters["installments"]
    has_installments = bool(current_filters["has_installments"])
    interest_free = bool(current_filters["interest_free"])

    cleaned_query = current_filters["cleaned_query"]
    search_terms = list(current_filters["search_terms"])
    raw_query = question

    if (
        not category
        and not search_terms
        and _can_use_standalone_context(question, current_filters, is_followup)
    ):
        cleaned_query = standalone_filters["cleaned_query"]
        search_terms = list(standalone_filters["search_terms"])

    return {
        "category": category,
        "raw_query": raw_query,
        "cleaned_query": cleaned_query,
        "search_terms": search_terms,
        "only_eminent": only_eminent,
        "exclude_eminent": exclude_eminent,
        "only_qr": current_filters["only_qr"],
        "only_nfc": current_filters["only_nfc"],
        "today_only": current_filters["today_only"],
        "every_day_only": current_filters["every_day_only"],
        "installments": installments,
        "has_installments": has_installments,
        "interest_free": interest_free,
        "explicit_eminent_black": bool(current_filters["explicit_eminent_black"]),
    }


def _can_use_standalone_context(
    question: str,
    current_filters: dict[str, Any],
    is_followup: bool,
) -> bool:
    if not is_followup:
        return False

    if any(
        [
            current_filters["category"],
            current_filters["only_eminent"],
            current_filters["exclude_eminent"],
            current_filters["only_qr"],
            current_filters["only_nfc"],
            current_filters["today_only"],
            current_filters["every_day_only"],
            current_filters["installments"],
            current_filters["has_installments"],
            current_filters["interest_free"],
            current_filters["search_terms"],
        ]
    ):
        return False

    normalized_question = _normalize_text(question)
    return any(
        re.search(rf"\b{re.escape(pattern)}\b", normalized_question)
        for pattern in REFERENCE_FOLLOWUP_PATTERNS
    )


def _should_show_categories_summary(filters: dict[str, Any]) -> bool:
    return not any(
        [
            filters["category"],
            filters["only_eminent"],
            filters["exclude_eminent"],
            filters["only_qr"],
            filters["only_nfc"],
            filters["today_only"],
            filters["every_day_only"],
            filters["installments"],
            filters["has_installments"],
            filters["interest_free"],
            filters["search_terms"],
        ]
    )


def _build_categories_answer() -> str:
    lines = [
        "🎁 Tengo beneficios para estas categorías:",
        "",
    ]

    for category in list_benefit_categories():
        emoji = CATEGORY_EMOJIS.get(category, "🎁")
        lines.append(f"{emoji} {category}")

    lines.extend(
        [
            "",
            "Podés pedirme, por ejemplo: *promos de gastronomía*, *beneficios en Frávega* o *beneficios exclusivos Eminent*.",
        ]
    )

    return "\n".join(lines)


def _build_results_answer(results: list[dict[str, Any]], filters: dict[str, Any]) -> str:
    header = _build_results_header(results, filters)
    blocks = [header, ""]

    for benefit in results[:5]:
        blocks.extend(_format_benefit_block(benefit))
        blocks.append("")

    return "\n".join(blocks).strip()


def _build_results_header(results: list[dict[str, Any]], filters: dict[str, Any]) -> str:
    category = filters["category"]
    segment = _eminent_label(filters)
    commerce_name = _unique_commerce_name(results)
    inferred_category = _unique_category_name(results)

    if filters["exclude_eminent"] and category:
        return f"🎁 Encontré estos beneficios de *{category}* que no son exclusivos *{segment}*:"

    if filters["exclude_eminent"]:
        return f"🎁 Encontré estos beneficios que no son exclusivos *{segment}*:"

    if filters["only_eminent"] and category:
        return f"💎 Encontré estos beneficios de *{category}* para *{segment}*:"

    if filters["only_eminent"]:
        return f"💎 Encontré estos beneficios exclusivos para *{segment}*:"

    if filters["only_qr"] and category:
        return f"📲 Encontré estos beneficios con *Pago QR* en *{category}*:"

    if filters["only_qr"]:
        return "📲 Encontré estos beneficios con *Pago QR*:"

    if filters["only_nfc"] and category:
        return f"📲 Encontré estos beneficios con *Pago NFC* en *{category}*:"

    if filters["only_nfc"]:
        return "📲 Encontré estos beneficios con *Pago NFC*:"

    if filters["installments"] and filters["interest_free"] and category:
        return (
            f"💳 Encontré estas promociones de *{category}* con "
            f"*{filters['installments']} cuotas sin interés*:"
        )

    if filters["installments"] and filters["interest_free"]:
        return f"💳 Encontré estas promociones con *{filters['installments']} cuotas sin interés*:"

    if filters["installments"] and category:
        return f"💳 Encontré estas promociones de *{category}* con *{filters['installments']} cuotas*:"

    if filters["installments"]:
        return f"💳 Encontré estas promociones con *{filters['installments']} cuotas*:"

    if filters["has_installments"] and filters["interest_free"] and category:
        return f"💳 Encontré estas promociones de *{category}* en *cuotas sin interés*:"

    if filters["has_installments"] and filters["interest_free"]:
        return "💳 Encontré estas promociones en *cuotas sin interés*:"

    if filters["has_installments"] and category:
        return f"💳 Encontré estas promociones de *{category}* en *cuotas*:"

    if filters["has_installments"]:
        return "💳 Encontré estas promociones en *cuotas*:"

    if filters["today_only"] and category:
        return f"🗓️ Encontré estos beneficios de *{category}* disponibles *hoy*:"

    if filters["today_only"]:
        return "🗓️ Encontré estos beneficios disponibles *hoy*:"

    if filters["every_day_only"] and category:
        return f"✅ Encontré estos beneficios de *{category}* para usar *todos los días*:"

    if filters["every_day_only"]:
        return "✅ Encontré estos beneficios para usar *todos los días*:"

    if commerce_name:
        return f"🔎 Encontré estos beneficios en *{commerce_name}*:"

    if inferred_category:
        return f"🎁 Encontré estos beneficios de *{inferred_category}*:"

    if category:
        return f"🎁 Encontré estos beneficios de *{category}*:"

    return "🔎 Encontré estos beneficios:"


def _unique_commerce_name(results: list[dict[str, Any]]) -> str | None:
    commerce_names = {
        str(result.get("comercio") or "").strip()
        for result in results
        if str(result.get("comercio") or "").strip()
    }

    if len(commerce_names) == 1:
        return next(iter(commerce_names))

    return None


def _unique_category_name(results: list[dict[str, Any]]) -> str | None:
    categories = {
        str(result.get("categoria") or "").strip()
        for result in results
        if str(result.get("categoria") or "").strip()
    }

    if len(categories) == 1:
        return next(iter(categories))

    return None


def _format_benefit_block(benefit: dict[str, Any]) -> list[str]:
    category = str(benefit.get("categoria") or "").strip()
    emoji = CATEGORY_EMOJIS.get(category, "🎁")
    commerce = str(benefit.get("comercio") or "Beneficio").strip()
    benefit_text = str(benefit.get("beneficio") or "").strip()

    details = [
        _format_days(benefit.get("dias")),
        f"💳 {_format_payment_methods(benefit.get('mediosDePago') or [])}",
    ]

    if benefit.get("exclusivoEminent"):
        details.append("💎 Exclusivo Eminent")

    if benefit.get("pagoQR"):
        details.append("📲 Pago QR")

    if benefit.get("pagoNFC"):
        details.append("📲 Pago NFC")

    if benefit.get("proximamente"):
        details.append("⏳ Próximamente")

    return [
        f"{emoji} *{commerce}* — {benefit_text}",
        f"🔹 {' | '.join(details)}",
    ]


def _format_days(days: Any) -> str:
    value = str(days or "").strip()
    return value or "Sin días informados"


def _format_payment_methods(methods: list[str]) -> str:
    clean_methods = [str(method).strip() for method in methods if str(method).strip()]

    if not clean_methods:
        return "Medios no informados"

    normalized = {_normalize_text(method) for method in clean_methods}
    if normalized == {"credito", "debito"}:
        return "Crédito y débito"

    if len(clean_methods) == 1:
        return clean_methods[0]

    return f"{', '.join(clean_methods[:-1])} y {clean_methods[-1]}"


def _build_no_results_answer(filters: dict[str, Any]) -> str:
    if filters["only_eminent"]:
        segment = _eminent_label(filters)
        return (
            f"🔎 No encontré beneficios exclusivos {segment} en las promociones disponibles ahora. "
            "Podés probar por categoría, por ejemplo Indumentaria, Viajes o Gastronomía."
        )

    if filters["only_nfc"]:
        return (
            "🔎 No encontré promociones con pago NFC en las promociones disponibles ahora. "
            "Sí podés consultar beneficios por QR, categoría o comercio."
        )

    if filters["installments"] and filters["interest_free"]:
        return (
            f"🔎 No encontré promociones con {filters['installments']} cuotas sin interés "
            "en las promociones disponibles ahora."
        )

    if filters["installments"]:
        return (
            f"🔎 No encontré promociones con {filters['installments']} cuotas "
            "en las promociones disponibles ahora."
        )

    if filters["has_installments"] and filters["interest_free"]:
        return "🔎 No encontré promociones en cuotas sin interés en las promociones disponibles ahora."

    if filters["has_installments"]:
        return "🔎 No encontré promociones en cuotas en las promociones disponibles ahora."

    if filters["only_qr"]:
        return "🔎 No encontré promociones con pago QR en las promociones disponibles ahora."

    if filters["category"]:
        return (
            f"🔎 No encontré beneficios para {filters['category']} en las promociones disponibles ahora. "
            "Si querés, también puedo buscar por un comercio puntual, por ejemplo *Frávega* o *Rappi*."
        )

    if filters["today_only"]:
        return "🔎 No encontré beneficios disponibles hoy en las promociones disponibles ahora."

    if filters["every_day_only"]:
        return "🔎 No encontré beneficios para usar todos los días en las promociones disponibles ahora."

    qualifier = _describe_filters(filters)
    return f"🔎 No encontré beneficios para {qualifier} en las promociones disponibles ahora."


def _describe_filters(filters: dict[str, Any]) -> str:
    if filters["category"] and filters["exclude_eminent"]:
        return f"*{filters['category']}* que no sea exclusivo Eminent"

    if filters["category"]:
        return f"*{filters['category']}*"

    if filters["exclude_eminent"]:
        return "beneficios no exclusivos Eminent"

    if filters["only_eminent"]:
        return f"*{_eminent_label(filters)}*"

    if filters["only_qr"]:
        return "*Pago QR*"

    if filters["only_nfc"]:
        return "*Pago NFC*"

    if filters["installments"] and filters["interest_free"]:
        return f"*{filters['installments']} cuotas sin interés*"

    if filters["installments"]:
        return f"*{filters['installments']} cuotas*"

    if filters["has_installments"] and filters["interest_free"]:
        return "*cuotas sin interés*"

    if filters["has_installments"]:
        return "*cuotas*"

    if filters["today_only"]:
        return "*hoy*"

    if filters["every_day_only"]:
        return "*todos los días*"

    if filters["search_terms"]:
        return f"ese criterio (*{' '.join(filters['search_terms'])}*)"

    return "ese criterio"


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


def _eminent_label(filters: dict[str, Any]) -> str:
    if filters.get("explicit_eminent_black"):
        return get_benefits_segment()
    return "Eminent"
