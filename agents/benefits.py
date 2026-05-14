from __future__ import annotations

from typing import Any

from agents.state import AgentState
from observability.logger import log_step
from tools.benefits_mock import (
    get_benefits_segment,
    infer_benefits_filters,
    list_benefit_categories,
    search_benefits,
)


CATEGORY_EMOJIS = {
    "Supermercados": "🛒",
    "Gastronomía": "🍽️",
    "Indumentaria": "👕",
    "Electrónica": "📱",
    "Hogar": "🏠",
}


def benefits_node(state: AgentState) -> AgentState:
    question = (state.get("standalone_question") or state.get("question") or "").strip()
    filters = infer_benefits_filters(question)

    log_step(
        "BENEFITS",
        "Consulta de beneficios interpretada",
        {
            "category": filters["category"],
            "only_eminent": filters["only_eminent"],
            "only_qr": filters["only_qr"],
            "only_nfc": filters["only_nfc"],
            "today_only": filters["today_only"],
            "every_day_only": filters["every_day_only"],
            "search_terms": filters["search_terms"],
        },
    )

    if _should_show_categories_summary(filters):
        answer = _build_categories_answer()
        results: list[dict[str, Any]] = []
    else:
        results = search_benefits(
            category=filters["category"],
            query=question,
            only_eminent=filters["only_eminent"],
            only_qr=filters["only_qr"],
            only_nfc=filters["only_nfc"],
            today_only=filters["today_only"],
            every_day_only=filters["every_day_only"],
            limit=5,
        )

        if results:
            answer = _build_results_answer(results, filters)
        else:
            answer = _build_no_results_answer(filters)

    return {
        **state,
        "route": "benefits",
        "tool_name": "benefits_mock",
        "tool_input": {
            "question": question,
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


def _should_show_categories_summary(filters: dict[str, Any]) -> bool:
    return not any(
        [
            filters["category"],
            filters["only_eminent"],
            filters["only_qr"],
            filters["only_nfc"],
            filters["today_only"],
            filters["every_day_only"],
            filters["search_terms"],
        ]
    )


def _build_categories_answer() -> str:
    lines = [
        "🎁 Tengo beneficios mockeados para estas categorías:",
        "",
    ]

    for category in list_benefit_categories():
        emoji = CATEGORY_EMOJIS.get(category, "🎁")
        lines.append(f"• {emoji} {category}")

    lines.extend(
        [
            "",
            "Podés pedirme, por ejemplo: *promos de gastronomía* o *beneficios exclusivos Eminent*.",
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
    segment = get_benefits_segment()
    commerce_name = _unique_commerce_name(results)

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

    if filters["today_only"] and category:
        return f"📅 Encontré estos beneficios de *{category}* disponibles *hoy*:"

    if filters["today_only"]:
        return "📅 Encontré estos beneficios disponibles *hoy*:"

    if filters["every_day_only"] and category:
        return f"✅ Encontré estos beneficios de *{category}* para usar *todos los días*:"

    if filters["every_day_only"]:
        return "✅ Encontré estos beneficios para usar *todos los días*:"

    if commerce_name:
        return f"🔎 Encontré estos beneficios en *{commerce_name}*:"

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


def _format_benefit_block(benefit: dict[str, Any]) -> list[str]:
    category = str(benefit.get("categoria") or "").strip()
    emoji = CATEGORY_EMOJIS.get(category, "🎁")
    commerce = str(benefit.get("comercio") or "Beneficio").strip()
    benefit_text = str(benefit.get("beneficio") or "").strip()

    details = [
        f"📅 {_format_days(benefit.get('dias'))}",
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
        " | ".join(details),
    ]


def _format_days(days: Any) -> str:
    value = str(days or "").strip()
    return value or "Sin días informados"


def _format_payment_methods(methods: list[str]) -> str:
    clean_methods = [str(method).strip() for method in methods if str(method).strip()]

    if not clean_methods:
        return "Medios no informados"

    normalized = {method.lower() for method in clean_methods}
    if normalized == {"credito", "debito"} or normalized == {"crédito", "débito"}:
        return "Crédito y débito"

    if len(clean_methods) == 1:
        return clean_methods[0]

    return f"{', '.join(clean_methods[:-1])} y {clean_methods[-1]}"


def _build_no_results_answer(filters: dict[str, Any]) -> str:
    categories = ", ".join(list_benefit_categories())
    qualifier = _describe_filters(filters)

    return (
        f"🔎 No encontré beneficios mock para {qualifier}. "
        f"Si querés, podés consultar por categoría: {categories}."
    )


def _describe_filters(filters: dict[str, Any]) -> str:
    if filters["category"]:
        return f"*{filters['category']}*"

    if filters["only_eminent"]:
        return "*Eminent Black*"

    if filters["only_qr"]:
        return "*Pago QR*"

    if filters["only_nfc"]:
        return "*Pago NFC*"

    if filters["today_only"]:
        return "*hoy*"

    if filters["every_day_only"]:
        return "*todos los días*"

    if filters["search_terms"]:
        return f"ese criterio (*{' '.join(filters['search_terms'])}*)"

    return "ese criterio"
