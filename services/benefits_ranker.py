from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any

from services.benefits_intelligence import extract_benefits_intent


def infer_benefits_search_context(
    query: str | None,
    *,
    intent_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return intent_context or extract_benefits_intent(query, llm=None)


def rank_locales(
    locales: list[dict[str, Any]],
    query: str | None,
    *,
    intent_context: dict[str, Any] | None = None,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    context = infer_benefits_search_context(query, intent_context=intent_context)
    ranked_locales: list[tuple[tuple[float, float, str, int], dict[str, Any]]] = []

    for index, local in enumerate(locales):
        local_copy = dict(local)
        score = _score_local_relevance(local_copy, context)
        distance = _safe_float(local_copy.get("distance_km"))
        distance_key = distance if distance is not None else float("inf")
        brand_key = _normalize_text(local_copy.get("brand") or local_copy.get("nombre") or "")

        local_copy["ranking_score"] = round(score, 4)
        ranked_locales.append(((-score, distance_key, brand_key, index), local_copy))

    ranked_locales.sort(key=lambda item: item[0])
    ordered = [local for _, local in ranked_locales]

    if max_candidates is not None:
        return _select_diverse_locales(ordered, max_candidates=max_candidates)

    return ordered


def rank_enriched_locales(
    locales: list[dict[str, Any]],
    query: str | None,
    *,
    intent_context: dict[str, Any] | None = None,
    max_results: int | None = None,
    reference_date: date | None = None,
) -> list[dict[str, Any]]:
    context = infer_benefits_search_context(query, intent_context=intent_context)
    today = reference_date or date.today()
    ranked_locales: list[tuple[tuple[float, float, str, int], dict[str, Any]]] = []

    for index, local in enumerate(locales):
        local_copy = dict(local)
        local_copy["promotions"] = [
            dict(promotion)
            for promotion in (local.get("promotions") or [])
            if isinstance(promotion, dict)
        ]

        primary_promotion = select_primary_promotion(
            local_copy,
            audience_hint=str(context.get("audience_hint") or "general"),
            reference_date=today,
        )
        score = _score_local_relevance(local_copy, context)
        score += _score_promotion_quality(
            primary_promotion,
            audience_hint=str(context.get("audience_hint") or "general"),
            reference_date=today,
        )
        score += _score_distance(local_copy.get("distance_km"))

        local_copy["selected_promotion"] = primary_promotion
        local_copy["has_additional_eminent_promotion"] = has_additional_eminent_promotion(
            local_copy,
            primary_promotion,
            reference_date=today,
        )
        local_copy["ranking_score"] = round(score, 4)
        audience_priority = 0
        if (
            str(context.get("audience_hint") or "general") != "eminent"
            and isinstance(primary_promotion, dict)
            and primary_promotion.get("is_eminent")
        ):
            audience_priority = 1

        distance = _safe_float(local_copy.get("distance_km"))
        distance_key = distance if distance is not None else float("inf")
        brand_key = _normalize_text(local_copy.get("brand") or local_copy.get("nombre") or "")
        ranked_locales.append(((audience_priority, -score, distance_key, brand_key, index), local_copy))

    ranked_locales.sort(key=lambda item: item[0])
    ordered = [local for _, local in ranked_locales]

    if max_results is not None:
        return _select_diverse_locales(ordered, max_candidates=max_results)

    return ordered


def select_primary_promotion(
    local: dict[str, Any],
    *,
    audience_hint: str = "general",
    reference_date: date | None = None,
) -> dict[str, Any] | None:
    today = reference_date or date.today()
    promotions = [
        promotion
        for promotion in (local.get("promotions") or [])
        if isinstance(promotion, dict) and _is_promotion_usable(promotion, today)
    ]

    if not promotions:
        return None

    if audience_hint != "eminent":
        mass_promotions = [promotion for promotion in promotions if not promotion.get("is_eminent")]
        if mass_promotions:
            return _sort_promotions(mass_promotions, audience_hint=audience_hint, today=today)[0]

    return _sort_promotions(promotions, audience_hint=audience_hint, today=today)[0]


def has_additional_eminent_promotion(
    local: dict[str, Any],
    primary_promotion: dict[str, Any] | None,
    *,
    reference_date: date | None = None,
) -> bool:
    today = reference_date or date.today()
    for promotion in local.get("promotions") or []:
        if not isinstance(promotion, dict) or promotion is primary_promotion:
            continue
        if not _is_promotion_usable(promotion, today):
            continue
        if promotion.get("is_eminent"):
            return True
    return False


def _score_local_relevance(local: dict[str, Any], context: dict[str, Any]) -> float:
    brand = str(local.get("brand") or local.get("nombre") or "").strip()
    category = str(local.get("category") or local.get("categoriaMarca") or "").strip()
    city = str(local.get("city") or local.get("localidad") or "").strip()
    address = str(local.get("address") or "").strip()

    normalized_brand = _normalize_text(brand)
    normalized_category = _normalize_text(category)
    searchable_text = " ".join(
        part
        for part in (
            normalized_brand,
            normalized_category,
            _normalize_text(city),
            _normalize_text(address),
            " ".join(_normalize_text(term) for term in context.get("store_type_hints") or []),
        )
        if part
    )

    score = 0.0

    for requested_category in context.get("category_candidates") or []:
        if _category_matches(category, requested_category):
            score += 120
            break

    for hint in context.get("brand_or_store_hints") or []:
        normalized_hint = _normalize_text(hint)
        if not normalized_hint:
            continue
        if normalized_hint in normalized_brand:
            score += 100
        elif normalized_hint in searchable_text:
            score += 35

    product_interest = _normalize_text(context.get("product_interest"))
    if product_interest and product_interest in searchable_text:
        score += 60

    recipient = _normalize_text(context.get("recipient"))
    if recipient and context.get("commercial_intent") == "gift_planning":
        score += 10

    query_terms = [
        token
        for token in _normalize_text(context.get("normalized_query")).split()
        if len(token) >= 4
    ]
    for term in query_terms:
        if term in normalized_brand:
            score += 40
        elif term in normalized_category:
            score += 18

    return score


def _score_promotion_quality(
    promotion: dict[str, Any] | None,
    *,
    audience_hint: str,
    reference_date: date,
) -> float:
    if not promotion:
        return 0.0

    score = 200.0

    discount_percent = _safe_float(promotion.get("discount_percent"))
    cashback_cap = _safe_float(promotion.get("cashback_cap"))

    if discount_percent is not None:
        score += min(discount_percent * 4, 140)

    if cashback_cap is not None:
        score += min(cashback_cap / 500, 70)

    if promotion.get("solo_today"):
        score += 30

    if _is_valid_today(promotion, reference_date):
        score += 20

    if promotion.get("is_eminent"):
        score += 15 if audience_hint == "eminent" else -35
    else:
        score += 25

    return score


def _score_distance(distance_value: Any) -> float:
    distance = _safe_float(distance_value)
    if distance is None:
        return 0.0

    return max(0.0, 40.0 - (distance * 4.0))


def _select_diverse_locales(
    ordered_locales: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    limit = max(1, max_candidates)
    primary_selection: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    seen_brands: set[str] = set()

    for local in ordered_locales:
        normalized_brand = _normalize_text(local.get("brand") or local.get("nombre") or "")
        if normalized_brand and normalized_brand not in seen_brands:
            seen_brands.add(normalized_brand)
            primary_selection.append(local)
        else:
            overflow.append(local)

        if len(primary_selection) >= limit:
            return primary_selection[:limit]

    for local in overflow:
        primary_selection.append(local)
        if len(primary_selection) >= limit:
            break

    return primary_selection[:limit]


def _sort_promotions(
    promotions: list[dict[str, Any]],
    *,
    audience_hint: str,
    today: date,
) -> list[dict[str, Any]]:
    def sort_key(promotion: dict[str, Any]) -> tuple[int, int, float, float, float]:
        return (
            0 if _is_valid_today(promotion, today) else 1,
            0 if (audience_hint == "eminent" or not promotion.get("is_eminent")) else 1,
            -(_safe_float(promotion.get("discount_percent")) or 0.0),
            -(_safe_float(promotion.get("cashback_cap")) or 0.0),
            -(1.0 if promotion.get("solo_today") else 0.0),
        )

    return sorted(promotions, key=sort_key)


def _is_promotion_usable(promotion: dict[str, Any], today: date) -> bool:
    if promotion.get("coming_soon"):
        return False

    valid_from = _parse_date(promotion.get("valid_from"))
    valid_to = _parse_date(promotion.get("valid_to"))

    if valid_from and today < valid_from:
        return False

    if valid_to and today > valid_to:
        return False

    return True


def _is_valid_today(promotion: dict[str, Any], today: date) -> bool:
    valid_from = _parse_date(promotion.get("valid_from"))
    valid_to = _parse_date(promotion.get("valid_to"))

    if valid_from and today < valid_from:
        return False

    if valid_to and today > valid_to:
        return False

    return True


def _parse_date(value: Any) -> date | None:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None

    for parser in (
        lambda text: datetime.strptime(text, "%d/%m/%Y").date(),
        lambda text: datetime.strptime(text, "%Y-%m-%d").date(),
    ):
        try:
            return parser(raw_value)
        except ValueError:
            continue

    return None


def _category_matches(local_category: str, target_category: str) -> bool:
    normalized_local_category = _normalize_text(local_category)
    normalized_target_category = _normalize_text(target_category)
    if not normalized_local_category or not normalized_target_category:
        return False

    return (
        normalized_local_category == normalized_target_category
        or normalized_target_category in normalized_local_category
        or normalized_local_category in normalized_target_category
    )


def _normalize_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    lowered = without_accents.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
