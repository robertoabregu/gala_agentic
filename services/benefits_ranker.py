from __future__ import annotations

import re
import unicodedata
from typing import Any


CATEGORY_KEYWORDS = {
    "Supermercados": (
        "super",
        "supermercado",
        "supermercados",
        "dia",
        "coto",
        "carrefour",
        "jumbo",
        "vea",
        "disco",
        "changomas",
    ),
    "Gastronomía": (
        "gastronomia",
        "comida",
        "delivery",
        "restaurante",
        "restaurantes",
        "hamburguesa",
        "hamburguesas",
        "cafe",
        "cafeteria",
        "helado",
        "pizza",
    ),
    "Indumentaria": (
        "ropa",
        "indumentaria",
        "zapatilla",
        "zapatillas",
        "zapas",
        "calzado",
        "vestimenta",
        "moda",
        "deportivo",
        "deportiva",
        "deportivas",
        "sport",
    ),
    "Electrónica": (
        "electronica",
        "tecnologia",
        "celular",
        "celulares",
        "notebook",
        "tv",
        "televisor",
        "electro",
    ),
    "Hogar": (
        "hogar",
        "casa",
        "muebles",
        "deco",
        "decoracion",
    ),
}

PRODUCT_HINTS = (
    {
        "name": "sportswear",
        "keywords": (
            "zapatilla",
            "zapatillas",
            "zapas",
            "botines",
            "camiseta",
            "deportiva",
            "deportivas",
            "running",
            "gym",
            "gimnasio",
        ),
        "preferred_categories": ("Indumentaria",),
        "preferred_brands": (
            "nike",
            "adidas",
            "puma",
            "reebok",
            "fila",
            "topper",
            "new balance",
            "under armour",
            "sportline",
            "dexter",
        ),
    },
)

STOPWORDS = {
    "a",
    "al",
    "algo",
    "beneficio",
    "beneficios",
    "cerca",
    "con",
    "de",
    "del",
    "descuento",
    "descuentos",
    "el",
    "en",
    "hay",
    "la",
    "las",
    "local",
    "locales",
    "los",
    "me",
    "mi",
    "mis",
    "papá",
    "papa",
    "para",
    "por",
    "promo",
    "promos",
    "promocion",
    "promociones",
    "que",
    "quiero",
    "su",
    "sus",
    "un",
    "una",
    "unos",
    "unas",
}


def infer_benefits_search_context(query: str | None) -> dict[str, Any]:
    normalized_query = _normalize_text(query)
    category = _detect_category(normalized_query)
    product_hint = _detect_product_hint(normalized_query)
    preferred_categories = []

    if category:
        preferred_categories.append(category)

    for preferred_category in product_hint.get("preferred_categories", ()):
        if preferred_category not in preferred_categories:
            preferred_categories.append(preferred_category)

    return {
        "normalized_query": normalized_query,
        "mentioned_category": category,
        "product_hint": product_hint.get("name"),
        "preferred_categories": preferred_categories,
        "preferred_brand_terms": list(product_hint.get("preferred_brands", ())),
        "query_terms": _extract_query_terms(normalized_query),
    }


def rank_locales(
    locales: list[dict[str, Any]],
    query: str | None,
    *,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    context = infer_benefits_search_context(query)
    ranked_locales: list[tuple[tuple[float, float, str, int], dict[str, Any]]] = []

    for index, local in enumerate(locales):
        local_copy = dict(local)
        score = _score_local(local_copy, context)
        distance = _safe_float(local_copy.get("distance_km"))
        distance_key = distance if distance is not None else float("inf")
        brand_key = str(local_copy.get("brand") or local_copy.get("nombre") or "").strip().lower()

        local_copy["ranking_score"] = score
        ranked_locales.append(((-score, distance_key, brand_key, index), local_copy))

    ranked_locales.sort(key=lambda item: item[0])
    ordered = [local for _, local in ranked_locales]

    if max_candidates is not None:
        return ordered[: max(1, max_candidates)]

    return ordered


def _score_local(local: dict[str, Any], context: dict[str, Any]) -> float:
    brand = str(local.get("brand") or local.get("nombre") or "").strip()
    category = str(local.get("category") or local.get("categoriaMarca") or "").strip()
    city = str(local.get("city") or local.get("localidad") or "").strip()

    normalized_brand = _normalize_text(brand)
    normalized_category = _normalize_text(category)
    searchable_text = " ".join(
        part
        for part in (
            normalized_brand,
            normalized_category,
            _normalize_text(city),
        )
        if part
    )

    score = 0.0

    mentioned_category = context.get("mentioned_category")
    if mentioned_category and _category_matches(category, mentioned_category):
        score += 120

    for preferred_category in context.get("preferred_categories") or []:
        if _category_matches(category, preferred_category):
            score += 80
            break

    normalized_query = str(context.get("normalized_query") or "")
    if normalized_brand and normalized_brand in normalized_query:
        score += 140

    query_terms = context.get("query_terms") or []
    for term in query_terms:
        if len(term) < 4:
            continue
        if term in normalized_brand:
            score += 45
        elif term in searchable_text:
            score += 15

    for preferred_brand in context.get("preferred_brand_terms") or []:
        normalized_preferred_brand = _normalize_text(preferred_brand)
        if normalized_preferred_brand and normalized_preferred_brand in normalized_brand:
            score += 70

    return score


def _detect_category(normalized_query: str) -> str | None:
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(_contains_term(normalized_query, keyword) for keyword in keywords):
            return category
    return None


def _detect_product_hint(normalized_query: str) -> dict[str, Any]:
    for hint in PRODUCT_HINTS:
        if any(_contains_term(normalized_query, keyword) for keyword in hint["keywords"]):
            return hint
    return {}


def _extract_query_terms(normalized_query: str) -> list[str]:
    return [
        token
        for token in normalized_query.split()
        if len(token) >= 3 and token not in STOPWORDS
    ]


def _category_matches(local_category: str, target_category: str) -> bool:
    normalized_local_category = _normalize_text(local_category)
    normalized_target_category = _normalize_text(target_category)

    if not normalized_local_category or not normalized_target_category:
        return False

    if normalized_local_category == normalized_target_category:
        return True

    keywords = CATEGORY_KEYWORDS.get(target_category, ())
    return any(_contains_term(normalized_local_category, keyword) for keyword in keywords)


def _contains_term(text: str, term: str) -> bool:
    normalized_text = f" {_normalize_text(text)} "
    normalized_term = f" {_normalize_text(term)} "
    return normalized_term in normalized_text


def _normalize_text(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
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
