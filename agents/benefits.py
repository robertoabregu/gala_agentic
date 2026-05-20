from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from agents.state import AgentState
from observability.logger import log_step
from services.benefits_ranker import infer_benefits_search_context, rank_locales
from tools.benefits_location_api import (
    enrich_local_with_detail,
    get_local_promotions_detail,
    get_nearby_locales,
)


CATEGORY_HEADERS = {
    "Supermercados": "🛒 Encontré estos supermercados cerca tuyo con beneficios Galicia:",
    "Gastronomía": "🍽️ Encontré estos locales gastronómicos cerca tuyo con beneficios Galicia:",
    "Indumentaria": "👟 Encontré estos locales de indumentaria cerca tuyo con beneficios Galicia:",
    "Electrónica": "💻 Encontré estos locales de electrónica cerca tuyo con beneficios Galicia:",
    "Hogar": "🏠 Encontré estos locales para el hogar cerca tuyo con beneficios Galicia:",
}


def benefits_node(state: AgentState) -> AgentState:
    question = (state.get("question") or "").strip()
    standalone_question = (state.get("standalone_question") or question).strip()
    query = standalone_question or question

    user_location = state.get("user_location") or {}
    latitude = _safe_float(user_location.get("latitude"))
    longitude = _safe_float(user_location.get("longitude"))

    if latitude is None or longitude is None:
        log_step("BENEFITS", "Falta ubicacion para consultar beneficios cercanos")
        return {
            **state,
            "route": "benefits",
            "tool_name": "benefits_location_api",
            "tool_input": {
                "question": question,
                "standalone_question": standalone_question,
            },
            "tool_output": {
                "results_count": 0,
                "results": [],
            },
            "answer": (
                "📍 Para buscar beneficios cerca tuyo necesito que me compartas tu ubicación "
                "actual desde WhatsApp."
            ),
            "topic": "beneficios",
            "needs_clarification": True,
            "missing_fields": ["user_location"],
            "pending_route": "benefits",
            "error": None,
        }

    max_results = _get_int_env("BENEFITS_MAX_RESULTS", 5, minimum=1)
    detail_max_candidates = _get_int_env("BENEFITS_DETAIL_MAX_CANDIDATES", 8, minimum=1)

    try:
        nearby_locales = get_nearby_locales(latitude, longitude)
        if not nearby_locales:
            return {
                **state,
                "route": "benefits",
                "tool_name": "benefits_location_api",
                "tool_input": {
                    "question": question,
                    "standalone_question": standalone_question,
                    "latitude": latitude,
                    "longitude": longitude,
                },
                "tool_output": {
                    "results_count": 0,
                    "results": [],
                },
                "answer": (
                    "📍 No encontré locales con beneficios cerca tuyo en este momento. "
                    "Si querés, probá con otro rubro o de nuevo más tarde."
                ),
                "topic": "beneficios",
                "needs_clarification": False,
                "missing_fields": [],
                "pending_route": "",
                "error": None,
            }

        ranked_locales = rank_locales(nearby_locales, query=query)
        top_candidates = ranked_locales[:detail_max_candidates]
        enriched_candidates: list[dict[str, Any]] = []
        detail_errors = 0
        detail_successes = 0

        for local in top_candidates:
            try:
                local_detail = get_local_promotions_detail(local["local_id"])
                detail_successes += 1
                enriched_candidates.append(enrich_local_with_detail(local, local_detail))
            except Exception as exc:
                detail_errors += 1
                log_step(
                    "BENEFITS",
                    "No se pudo obtener el detalle de un local",
                    {
                        "local_id": local.get("local_id"),
                        "brand": local.get("brand"),
                        "error": str(exc),
                    },
                )
                enriched_candidates.append(enrich_local_with_detail(local, None))

        search_context = infer_benefits_search_context(query)
        answer = _build_answer(
            locals_with_details=enriched_candidates,
            search_context=search_context,
            max_results=max_results,
            detail_successes=detail_successes,
        )

        log_step(
            "BENEFITS",
            "Beneficios cercanos procesados",
            {
                "query": question,
                "nearby_locales": len(nearby_locales),
                "detail_candidates": len(top_candidates),
                "detail_successes": detail_successes,
                "detail_errors": detail_errors,
            },
        )

        return {
            **state,
            "route": "benefits",
            "tool_name": "benefits_location_api",
            "tool_input": {
                "question": question,
                "standalone_question": standalone_question,
                "latitude": latitude,
                "longitude": longitude,
                "detail_max_candidates": detail_max_candidates,
                "max_results": max_results,
            },
            "tool_output": {
                "results_count": min(len(enriched_candidates), max_results),
                "results": enriched_candidates[:max_results],
                "nearby_locales_count": len(nearby_locales),
                "detail_candidates_count": len(top_candidates),
                "detail_successes": detail_successes,
            },
            "answer": answer,
            "topic": "beneficios",
            "needs_clarification": False,
            "missing_fields": [],
            "pending_route": "",
            "error": None,
        }
    except Exception as exc:
        log_step(
            "BENEFITS",
            "Error consultando beneficios cercanos",
            {
                "question": question,
                "latitude": latitude,
                "longitude": longitude,
                "error": str(exc),
            },
        )
        return {
            **state,
            "route": "benefits",
            "tool_name": "benefits_location_api",
            "tool_input": {
                "question": question,
                "standalone_question": standalone_question,
                "latitude": latitude,
                "longitude": longitude,
            },
            "tool_output": {
                "results_count": 0,
                "results": [],
            },
            "answer": (
                "🔎 No pude consultar los beneficios de Galicia en este momento. "
                "Si querés, probá de nuevo en un ratito."
            ),
            "topic": "beneficios",
            "needs_clarification": False,
            "missing_fields": [],
            "pending_route": "",
            "error": str(exc),
        }


def _build_answer(
    *,
    locals_with_details: list[dict[str, Any]],
    search_context: dict[str, Any],
    max_results: int,
    detail_successes: int,
) -> str:
    promo_ready_locals = [
        local
        for local in locals_with_details
        if _select_primary_promotion(local) is not None
    ]

    if promo_ready_locals:
        lines = [_build_header(search_context), ""]
        selected_locals = promo_ready_locals[:max_results]

        for index, local in enumerate(selected_locals, start=1):
            lines.extend(_format_promo_local(index, local))
            lines.append("")

        first_local = selected_locals[0]
        if first_local.get("brand"):
            lines.append(
                f"Te conviene arrancar por *{first_local['brand']}* porque es el más cercano."
            )

        return "\n".join(lines).strip()

    lines = []
    if detail_successes == 0:
        lines.append(
            "📍 Encontré locales cerca tuyo, pero ahora no pude confirmar el detalle de las promociones."
        )
        lines.append("Te paso igual los más cercanos:")
    else:
        lines.append(
            "📍 Encontré locales cerca tuyo, pero no vi promociones vigentes o claras en los mejores candidatos."
        )
        lines.append("Te paso igual las opciones más cercanas:")

    lines.append("")

    for index, local in enumerate(locals_with_details[:max_results], start=1):
        lines.extend(_format_nearby_local(index, local))
        lines.append("")

    lines.append("Si querés, también puedo probar con otro rubro o con un comercio puntual.")
    return "\n".join(lines).strip()


def _build_header(search_context: dict[str, Any]) -> str:
    category = search_context.get("mentioned_category")
    if category and category in CATEGORY_HEADERS:
        return CATEGORY_HEADERS[category]
    return "📍 Encontré estos locales cerca tuyo con beneficios Galicia:"


def _format_promo_local(index: int, local: dict[str, Any]) -> list[str]:
    primary_promotion = _select_primary_promotion(local)
    if primary_promotion is None:
        return _format_nearby_local(index, local)

    brand = str(local.get("brand") or "Local adherido").strip()
    address = _best_address(local)
    distance_text = _format_distance(local.get("distance_km"))

    header = f"{index}. *{brand}*"
    if address:
        header = f"{header} — {address}"

    lines = [header]

    if distance_text:
        lines.append(f"📍 A {distance_text} aprox.")

    discount_percent = primary_promotion.get("discount_percent")
    if discount_percent is not None:
        lines.append(f"💳 {_format_discount(discount_percent)} de ahorro")

    days = str(primary_promotion.get("days") or "").strip()
    if days:
        lines.append(f"🗓️ {days}")

    cashback_cap = primary_promotion.get("cashback_cap")
    if cashback_cap is not None:
        lines.append(f"💰 Tope: {_format_money(cashback_cap)}")

    payment_summary = str(primary_promotion.get("payment_summary") or "").strip()
    if payment_summary:
        lines.append(f"💳 Medios: {payment_summary}")

    if _has_additional_eminent_promotion(local, primary_promotion):
        lines.append("💎 También tiene un beneficio adicional para clientes Eminent.")
    elif primary_promotion.get("is_eminent"):
        lines.append("💎 Aplica para clientes Eminent.")

    return lines


def _format_nearby_local(index: int, local: dict[str, Any]) -> list[str]:
    brand = str(local.get("brand") or "Local adherido").strip()
    address = _best_address(local)
    distance_text = _format_distance(local.get("distance_km"))

    header = f"{index}. *{brand}*"
    if address:
        header = f"{header} — {address}"

    lines = [header]
    if distance_text:
        lines.append(f"📍 A {distance_text} aprox.")
    return lines


def _select_primary_promotion(local: dict[str, Any]) -> dict[str, Any] | None:
    promotions = [
        promotion
        for promotion in (local.get("promotions") or [])
        if isinstance(promotion, dict) and not promotion.get("coming_soon")
    ]
    if not promotions:
        return None

    mass_promotions = [promotion for promotion in promotions if not promotion.get("is_eminent")]
    if mass_promotions:
        return _sort_promotions(mass_promotions)[0]

    return _sort_promotions(promotions)[0]


def _has_additional_eminent_promotion(
    local: dict[str, Any],
    primary_promotion: dict[str, Any],
) -> bool:
    for promotion in local.get("promotions") or []:
        if not isinstance(promotion, dict) or promotion is primary_promotion:
            continue
        if promotion.get("coming_soon"):
            continue
        if promotion.get("is_eminent"):
            return True
    return False


def _sort_promotions(promotions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(promotion: dict[str, Any]) -> tuple[int, float, float]:
        discount = _safe_float(promotion.get("discount_percent")) or 0.0
        cap = _safe_float(promotion.get("cashback_cap")) or 0.0
        return (0 if not promotion.get("is_eminent") else 1, -discount, -cap)

    return sorted(promotions, key=sort_key)


def _best_address(local: dict[str, Any]) -> str:
    address = str(local.get("address") or "").strip()
    if address:
        return address

    city = str(local.get("city") or "").strip()
    province = str(local.get("province") or "").strip()
    if city and province and _normalize_text(city) != _normalize_text(province):
        return f"{city}, {province}"
    return city or province


def _format_distance(distance_km: Any) -> str:
    distance = _safe_float(distance_km)
    if distance is None:
        return ""
    return f"{distance:.1f} km"


def _format_discount(value: int | float) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""
    if numeric.is_integer():
        return f"{int(numeric)}%"
    return f"{numeric:.1f}%"


def _format_money(value: int | float) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""

    if numeric.is_integer():
        integer_value = int(numeric)
        return f"${integer_value:,}".replace(",", ".")

    whole_part, decimal_part = f"{numeric:.2f}".split(".")
    formatted_whole = f"{int(whole_part):,}".replace(",", ".")
    if decimal_part == "00":
        return f"${formatted_whole}"
    return f"${formatted_whole},{decimal_part}"


def _get_int_env(name: str, default: int, *, minimum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    lowered = without_accents.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()
