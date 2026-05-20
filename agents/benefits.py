from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import unicodedata
from datetime import date
from typing import Any

from agents.state import AgentState
from observability.logger import log_step
from services.benefits_intelligence import extract_benefits_intent
from services.benefits_ranker import rank_enriched_locales, rank_locales
from tools.benefits_location_api import (
    enrich_local_with_detail,
    get_local_promotions_detail,
    get_nearby_locales,
)

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - fallback solo para tests sin dependencias
    ChatOpenAI = Any  # type: ignore[misc,assignment]


BENEFITS_RESPONSE_PROMPT = """
Sos un asistente bancario que responde por WhatsApp en español argentino.
Tu tarea es redactar una respuesta consultiva sobre beneficios cercanos usando SOLO el JSON recibido.

Reglas:
- No inventes promociones, direcciones, porcentajes, topes, días, vigencias ni medios de pago.
- Si un dato no viene, omitilo.
- No empieces con "Hola", "Buenas" ni saludos similares.
- No digas "mejores descuentos" ni afirmaciones que no se puedan justificar.
- Sí podés hablar de "opciones recomendadas" o "locales cercanos con beneficios Galicia".
- Si viene "response_hint" dentro de "active_occasion", podés usarlo al comienzo.
- Si "selected_promotion.is_eminent" es true, aclará que aplica para clientes Eminent.
- Si "has_additional_eminent_promotion" es true y la promo principal no es Eminent, aclará que también hay un beneficio adicional para clientes Eminent.
- Mostrá cada local en este estilo:
  1. *Nombre* — Dirección
  📍 A X km aprox.
  💳 XX% de ahorro
  🗓️ Días
  💰 Tope: $X
  💳 Medios: ...
- Si no hay promociones claras, explicalo sin inventar nada y listá igual las opciones cercanas.
- Cerrá con una recomendación breve y útil si se puede justificar con la categoría, el producto o la cercanía.
"""

LOCATION_PLACEHOLDER_PATTERNS = {
    "ubicacion compartida por whatsapp",
    "ubicación compartida por whatsapp",
}

CATEGORY_EMOJIS = {
    "Supermercados": "🛒",
    "Gastronomía": "🍽️",
    "Indumentaria": "👟",
    "Electrónica": "💻",
    "Hogar": "🏠",
    "Salud y Bienestar": "💄",
    "Juguetes": "🧸",
    "Mascotas": "🐾",
    "Librerías": "📚",
}


def benefits_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
) -> AgentState:
    question = (state.get("question") or "").strip()
    standalone_question = (state.get("standalone_question") or question).strip()
    memory = state.get("memory") or {}
    resolved_query = _resolve_benefits_query(
        question=question,
        standalone_question=standalone_question,
        memory=memory,
    )

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
                "resolved_query": resolved_query,
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
    detail_max_candidates = _get_int_env("BENEFITS_DETAIL_MAX_CANDIDATES", 5, minimum=1)
    detail_expansion_batch = _get_int_env("BENEFITS_DETAIL_EXPANSION_BATCH", 3, minimum=1)
    min_promo_results = _get_int_env("BENEFITS_MIN_PROMO_RESULTS", 3, minimum=1)

    try:
        intent_context = extract_benefits_intent(resolved_query, llm=llm)
        nearby_locales = get_nearby_locales(latitude, longitude)

        if not nearby_locales:
            return _build_no_locales_state(
                state=state,
                question=question,
                standalone_question=standalone_question,
                resolved_query=resolved_query,
                latitude=latitude,
                longitude=longitude,
                intent_context=intent_context,
            )

        ranked_candidates_pool = rank_locales(
            nearby_locales,
            query=resolved_query,
            intent_context=intent_context,
            max_candidates=detail_max_candidates + detail_expansion_batch,
        )

        initial_candidates = ranked_candidates_pool[:detail_max_candidates]
        expansion_candidates = ranked_candidates_pool[
            detail_max_candidates:detail_max_candidates + detail_expansion_batch
        ]
        expansion_used = False

        enriched_candidates, detail_successes, detail_errors = _fetch_enriched_candidates(
            initial_candidates
        )
        ranked_results = _rank_benefits_results(
            enriched_candidates=enriched_candidates,
            resolved_query=resolved_query,
            intent_context=intent_context,
            max_results=max_results,
        )

        if expansion_candidates and _count_useful_results(ranked_results) < min_promo_results:
            extra_candidates, extra_successes, extra_errors = _fetch_enriched_candidates(
                expansion_candidates
            )
            enriched_candidates.extend(extra_candidates)
            detail_successes += extra_successes
            detail_errors += extra_errors
            ranked_results = _rank_benefits_results(
                enriched_candidates=enriched_candidates,
                resolved_query=resolved_query,
                intent_context=intent_context,
                max_results=max_results,
            )
            expansion_used = True

        answer = _build_answer(
            query=resolved_query,
            ranked_results=ranked_results,
            intent_context=intent_context,
            detail_successes=detail_successes,
            llm=llm,
        )

        log_step(
            "BENEFITS",
            "Beneficios cercanos procesados",
            {
                "query": resolved_query,
                "nearby_locales": len(nearby_locales),
                "initial_detail_candidates": len(initial_candidates),
                "expanded_detail_candidates": len(expansion_candidates) if expansion_used else 0,
                "detail_successes": detail_successes,
                "detail_errors": detail_errors,
                "final_results": len(ranked_results),
            },
        )

        return {
            **state,
            "route": "benefits",
            "tool_name": "benefits_location_api",
            "tool_input": {
                "question": question,
                "standalone_question": standalone_question,
                "resolved_query": resolved_query,
                "latitude": latitude,
                "longitude": longitude,
                "detail_max_candidates": detail_max_candidates,
                "detail_expansion_batch": detail_expansion_batch,
                "min_promo_results": min_promo_results,
                "max_results": max_results,
            },
            "tool_output": {
                "results_count": len(ranked_results),
                "results": ranked_results,
                "nearby_locales_count": len(nearby_locales),
                "detail_candidates_count": len(initial_candidates) + (
                    len(expansion_candidates) if expansion_used else 0
                ),
                "initial_detail_candidates_count": len(initial_candidates),
                "expanded_detail_candidates_count": len(expansion_candidates) if expansion_used else 0,
                "detail_successes": detail_successes,
                "intent_context": intent_context,
                "active_occasion": intent_context.get("active_occasion"),
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
                "resolved_query": resolved_query,
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
                "resolved_query": resolved_query,
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


def _build_no_locales_state(
    *,
    state: AgentState,
    question: str,
    standalone_question: str,
    resolved_query: str,
    latitude: float,
    longitude: float,
    intent_context: dict[str, Any],
) -> AgentState:
    category_candidates = intent_context.get("category_candidates") or []
    if category_candidates:
        answer = (
            f"📍 No encontré locales cercanos con beneficios para *{category_candidates[0]}* "
            "en este momento. Si querés, puedo probar con otro rubro."
        )
    else:
        answer = (
            "📍 No encontré locales con beneficios cerca tuyo en este momento. "
            "Si querés, probá con otro rubro o de nuevo más tarde."
        )

    return {
        **state,
        "route": "benefits",
        "tool_name": "benefits_location_api",
        "tool_input": {
            "question": question,
            "standalone_question": standalone_question,
            "resolved_query": resolved_query,
            "latitude": latitude,
            "longitude": longitude,
        },
        "tool_output": {
            "results_count": 0,
            "results": [],
            "intent_context": intent_context,
        },
        "answer": answer,
        "topic": "beneficios",
        "needs_clarification": False,
        "missing_fields": [],
        "pending_route": "",
        "error": None,
    }


def _fetch_enriched_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    if not candidates:
        return [], 0, 0

    enriched_candidates_by_index: dict[int, dict[str, Any]] = {}
    detail_errors = 0
    detail_successes = 0

    max_workers = min(4, len(candidates)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(get_local_promotions_detail, local["local_id"]): (index, local)
            for index, local in enumerate(candidates)
        }
        for future in as_completed(future_map):
            index, local = future_map[future]
            try:
                local_detail = future.result()
                detail_successes += 1
                enriched_candidates_by_index[index] = enrich_local_with_detail(local, local_detail)
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
                enriched_candidates_by_index[index] = enrich_local_with_detail(local, None)

    enriched_candidates = [
        enriched_candidates_by_index[index]
        for index in sorted(enriched_candidates_by_index)
    ]
    return enriched_candidates, detail_successes, detail_errors


def _rank_benefits_results(
    *,
    enriched_candidates: list[dict[str, Any]],
    resolved_query: str,
    intent_context: dict[str, Any],
    max_results: int,
) -> list[dict[str, Any]]:
    return rank_enriched_locales(
        enriched_candidates,
        query=resolved_query,
        intent_context=intent_context,
        max_results=max_results,
        reference_date=date.today(),
    )


def _count_useful_results(ranked_results: list[dict[str, Any]]) -> int:
    return sum(
        1
        for local in ranked_results
        if isinstance(local.get("selected_promotion"), dict)
    )


def _build_answer(
    *,
    query: str,
    ranked_results: list[dict[str, Any]],
    intent_context: dict[str, Any],
    detail_successes: int,
    llm: ChatOpenAI | None,
) -> str:
    payload = {
        "query": query,
        "intent_context": {
            "intent": intent_context.get("intent"),
            "commercial_intent": intent_context.get("commercial_intent"),
            "product_interest": intent_context.get("product_interest"),
            "recipient": intent_context.get("recipient"),
            "occasion": intent_context.get("occasion"),
            "category_candidates": intent_context.get("category_candidates") or [],
            "store_type_hints": intent_context.get("store_type_hints") or [],
            "brand_or_store_hints": intent_context.get("brand_or_store_hints") or [],
            "audience_hint": intent_context.get("audience_hint") or "general",
            "active_occasion": intent_context.get("active_occasion"),
        },
        "detail_successes": detail_successes,
        "locals": [_serialize_local_for_response(local) for local in ranked_results],
    }

    if llm is not None:
        try:
            response = llm.invoke(
                [
                    ("system", BENEFITS_RESPONSE_PROMPT),
                    ("user", json.dumps(payload, ensure_ascii=False)),
                ]
            )
            answer = _strip_leading_greeting(
                _coerce_text_response(getattr(response, "content", ""))
            )
            if answer:
                return answer
        except Exception as exc:
            log_step("BENEFITS", "Fallo la redaccion LLM de benefits", {"error": str(exc)})

    return _build_deterministic_answer(
        ranked_results=ranked_results,
        intent_context=intent_context,
        detail_successes=detail_successes,
    )


def _build_deterministic_answer(
    *,
    ranked_results: list[dict[str, Any]],
    intent_context: dict[str, Any],
    detail_successes: int,
) -> str:
    promo_ready_results = [
        local for local in ranked_results if isinstance(local.get("selected_promotion"), dict)
    ]

    if not promo_ready_results:
        return _build_nearby_only_answer(
            ranked_results=ranked_results,
            detail_successes=detail_successes,
            intent_context=intent_context,
        )

    lines = [_build_header(intent_context), ""]

    for index, local in enumerate(promo_ready_results, start=1):
        lines.extend(_format_local_block(index, local))
        lines.append("")

    closing = _build_closing(promo_ready_results, intent_context)
    if closing:
        lines.append(closing)

    return "\n".join(lines).strip()


def _build_nearby_only_answer(
    *,
    ranked_results: list[dict[str, Any]],
    detail_successes: int,
    intent_context: dict[str, Any],
) -> str:
    lines = []
    if detail_successes == 0:
        lines.append(
            "📍 Encontré locales cerca tuyo, pero ahora no pude confirmar el detalle de las promociones."
        )
    else:
        lines.append(
            "📍 Encontré locales cerca tuyo, pero no vi promociones vigentes o claras en las opciones más relevantes."
        )

    active_occasion = intent_context.get("active_occasion") or {}
    if active_occasion.get("response_hint"):
        lines.append(str(active_occasion["response_hint"]).strip())

    lines.append("Te paso igual algunas opciones cercanas:")
    lines.append("")

    for index, local in enumerate(ranked_results, start=1):
        header = f"{index}. *{str(local.get('brand') or 'Local adherido').strip()}*"
        address = _best_address(local)
        if address:
            header = f"{header} — {address}"
        lines.append(header)
        distance = _format_distance(local.get("distance_km"))
        if distance:
            lines.append(f"📍 A {distance} aprox.")
        lines.append("")

    lines.append("Si querés, también puedo probar con otro rubro o con un comercio puntual.")
    return "\n".join(lines).strip()


def _build_header(intent_context: dict[str, Any]) -> str:
    active_occasion = intent_context.get("active_occasion") or {}
    response_hint = str(active_occasion.get("response_hint") or "").strip()
    primary_category = next(iter(intent_context.get("category_candidates") or []), None)
    product_interest = str(intent_context.get("product_interest") or "").strip()

    if response_hint:
        emoji = CATEGORY_EMOJIS.get(primary_category, "📍")
        detail = "opciones cercanas con beneficios Galicia"
        if product_interest:
            detail = f"opciones cercanas que pueden servir para {product_interest}"
        return f"{emoji} {response_hint}\nBusqué {detail}:"

    if intent_context.get("commercial_intent") == "gift_planning" and product_interest:
        emoji = CATEGORY_EMOJIS.get(primary_category, "🎁")
        return f"{emoji} Busqué opciones cercanas que pueden servir para regalar {product_interest}:"

    if primary_category:
        emoji = CATEGORY_EMOJIS.get(primary_category, "📍")
        category_label = primary_category.lower()
        return f"{emoji} Encontré estos locales de {category_label} cerca tuyo con beneficios Galicia:"

    return "📍 Encontré estos locales cercanos con beneficios Galicia:"


def _format_local_block(index: int, local: dict[str, Any]) -> list[str]:
    brand = str(local.get("brand") or "Local adherido").strip()
    address = _best_address(local)
    distance_text = _format_distance(local.get("distance_km"))
    promotion = local.get("selected_promotion") or {}

    header = f"{index}. *{brand}*"
    if address:
        header = f"{header} — {address}"

    lines = [header]
    if distance_text:
        lines.append(f"📍 A {distance_text} aprox.")

    discount_percent = promotion.get("discount_percent")
    if discount_percent is not None:
        lines.append(f"💳 {_format_discount(discount_percent)} de ahorro")

    days = str(promotion.get("days") or "").strip()
    if days:
        lines.append(f"🗓️ {days}")

    cashback_cap = promotion.get("cashback_cap")
    if cashback_cap is not None:
        lines.append(f"💰 Tope: {_format_money(cashback_cap)}")

    payment_text = str(
        promotion.get("payment_summary")
        or promotion.get("pay_legend")
        or ""
    ).strip()
    if payment_text:
        lines.append(f"💳 Medios: {payment_text}")

    if promotion.get("is_eminent"):
        lines.append("💎 Aplica para clientes Eminent.")
    elif local.get("has_additional_eminent_promotion"):
        lines.append("💎 También tiene un beneficio adicional para clientes Eminent.")

    return lines


def _build_closing(
    ranked_results: list[dict[str, Any]],
    intent_context: dict[str, Any],
) -> str:
    top_brands = [
        str(local.get("brand") or "").strip()
        for local in ranked_results[:2]
        if str(local.get("brand") or "").strip()
    ]
    unique_top_brands = []
    seen: set[str] = set()
    for brand in top_brands:
        normalized_brand = _normalize_text(brand)
        if normalized_brand in seen:
            continue
        seen.add(normalized_brand)
        unique_top_brands.append(brand)

    product_interest = str(intent_context.get("product_interest") or "").strip()
    store_type_hints = [str(hint).strip() for hint in intent_context.get("store_type_hints") or [] if str(hint).strip()]

    if intent_context.get("commercial_intent") == "gift_planning" and unique_top_brands:
        preferred_types = " / ".join(store_type_hints[:2]) if store_type_hints else "ese tipo de compra"
        if len(unique_top_brands) == 1:
            return (
                f"Para {product_interest or 'este tipo de regalo'}, miraría primero *{unique_top_brands[0]}* "
                f"porque parece una opción bien orientada a {preferred_types}."
            )
        return (
            f"Para {product_interest or 'este tipo de regalo'}, miraría primero "
            f"*{unique_top_brands[0]}* o *{unique_top_brands[1]}* porque están más orientados a {preferred_types}."
        )

    if not unique_top_brands:
        return ""

    if len(unique_top_brands) == 1:
        return f"Te conviene arrancar por *{unique_top_brands[0]}* porque es una de las opciones más cercanas."

    return (
        f"Te conviene arrancar por *{unique_top_brands[0]}* o *{unique_top_brands[1]}* "
        "porque quedaron entre las opciones más cercanas y relevantes."
    )


def _serialize_local_for_response(local: dict[str, Any]) -> dict[str, Any]:
    promotion = local.get("selected_promotion") or {}
    return {
        "brand": local.get("brand"),
        "category": local.get("category"),
        "address": _best_address(local) or None,
        "distance_km": _safe_float(local.get("distance_km")),
        "selected_promotion": {
            "discount_percent": promotion.get("discount_percent"),
            "cashback_cap": promotion.get("cashback_cap"),
            "days": promotion.get("days"),
            "payment_summary": promotion.get("payment_summary"),
            "pay_legend": promotion.get("pay_legend"),
            "is_eminent": bool(promotion.get("is_eminent")),
        }
        if promotion
        else None,
        "has_additional_eminent_promotion": bool(local.get("has_additional_eminent_promotion")),
    }


def _resolve_benefits_query(
    *,
    question: str,
    standalone_question: str,
    memory: dict[str, Any],
) -> str:
    pending_route = str(memory.get("pending_route") or "").strip()
    pending_query = str(memory.get("pending_query") or "").strip()
    if pending_route == "benefits" and pending_query:
        return pending_query

    candidate = standalone_question or question
    normalized_candidate = _normalize_text(candidate)

    if normalized_candidate in LOCATION_PLACEHOLDER_PATTERNS:
        memory_question = pending_query or str(memory.get("last_user_question") or "").strip()
        if memory_question:
            return memory_question

    return candidate or pending_query or str(memory.get("last_user_question") or "").strip() or question


def _coerce_text_response(content: Any) -> str:
    if isinstance(content, list):
        text = "\n".join(
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    else:
        text = str(content or "")

    return text.strip()


def _strip_leading_greeting(answer: str) -> str:
    cleaned = re.sub(
        r"^\W*(hola|buenas|buen dia|buenos dias|buenas tardes|buenas noches)\b[\W_]*",
        "",
        answer or "",
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


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


def _normalize_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    lowered = without_accents.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()
