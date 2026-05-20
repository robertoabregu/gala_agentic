from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from typing import Any

from observability.logger import log_step
from services.commercial_calendar import detect_explicit_occasion, get_active_commercial_context

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - fallback solo para tests sin dependencias
    ChatOpenAI = Any  # type: ignore[misc,assignment]


KNOWN_CATEGORIES = [
    "Supermercados",
    "Gastronomía",
    "Indumentaria",
    "Electrónica",
    "Hogar",
    "Salud y Bienestar",
    "Juguetes",
    "Mascotas",
    "Librerías",
    "Viajes",
    "Vehículos",
    "Otros",
]

CATEGORY_ALIASES = {
    "Supermercados": (
        "super",
        "supermercado",
        "supermercados",
        "compras del super",
        "compras del súper",
        "comida para casa",
        "dia",
        "jumbo",
        "carrefour",
        "coto",
        "vea",
        "disco",
        "changomas",
    ),
    "Gastronomía": (
        "comer",
        "almorzar",
        "cenar",
        "hamburguesa",
        "cafe",
        "café",
        "helado",
        "pizza",
        "restaurante",
        "restaurantes",
        "gastronomia",
        "gastronomía",
    ),
    "Indumentaria": (
        "ropa",
        "indumentaria",
        "zapatilla",
        "zapatillas",
        "calzado",
        "ropa deportiva",
        "deportes",
        "deportivo",
        "deportiva",
        "deportivas",
        "sport",
    ),
    "Electrónica": (
        "celular",
        "notebook",
        "televisor",
        "tv",
        "auriculares",
        "electronica",
        "electrónica",
        "tecnologia",
        "tecnología",
    ),
    "Juguetes": (
        "juguete",
        "juguetes",
        "regalo para chicos",
        "regalo para niños",
        "día del niño",
        "dia del nino",
    ),
    "Salud y Bienestar": (
        "perfume",
        "perfumes",
        "farmacia",
        "farmacias",
        "belleza",
        "peluqueria",
        "peluquería",
        "cosmética",
        "cosmetica",
    ),
    "Hogar": (
        "muebles",
        "colchon",
        "colchón",
        "decoracion",
        "decoración",
        "pintura",
        "hogar",
        "casa",
    ),
    "Librerías": (
        "libros",
        "libro",
        "utiles",
        "útiles",
        "estudiar",
        "libreria",
        "librerías",
        "librerias",
    ),
    "Mascotas": (
        "perro",
        "gato",
        "mascota",
        "mascotas",
        "pet",
    ),
    "Viajes": (
        "viaje",
        "viajes",
        "hotel",
        "vuelos",
        "vuelo",
        "turismo",
    ),
    "Vehículos": (
        "auto",
        "autos",
        "moto",
        "motos",
        "vehiculo",
        "vehículos",
        "vehiculos",
        "neumatico",
        "neumático",
    ),
}

PRODUCT_HINTS = [
    {
        "keywords": ("zapatilla", "zapatillas", "calzado", "botines", "ropa deportiva"),
        "product_interest": "zapatillas",
        "category_candidates": ["Indumentaria"],
        "store_type_hints": ["calzado", "deportes", "ropa deportiva"],
        "brand_or_store_hints": [
            "Dexter",
            "Moov",
            "Stock Center",
            "Nike",
            "Adidas",
            "Reebok",
            "Vans",
            "Sporting",
            "Open Sports",
        ],
    },
    {
        "keywords": ("supermercado", "supermercados", "super", "compras del super"),
        "product_interest": None,
        "category_candidates": ["Supermercados"],
        "store_type_hints": ["supermercado"],
        "brand_or_store_hints": ["DIA", "Jumbo", "Carrefour", "Coto", "Disco", "Vea"],
    },
    {
        "keywords": ("hamburguesa", "hamburguesas", "cafe", "café", "helado", "comer"),
        "product_interest": None,
        "category_candidates": ["Gastronomía"],
        "store_type_hints": ["gastronomía"],
        "brand_or_store_hints": [],
    },
]

RECIPIENT_HINTS = {
    "padre": ("papa", "papá", "padre", "mi viejo"),
    "madre": ("mama", "mamá", "madre"),
    "hijos": ("chicos", "niños", "ninos", "hijos", "nenes"),
    "pareja": ("pareja", "novio", "novia", "marido", "mujer"),
}

GIFT_PATTERNS = (
    "regalo",
    "regalar",
    "comprarle",
    "comprarles",
)

EMINENT_PATTERNS = (
    "eminent",
    "eminent black",
)

KNOWN_STORE_HINTS = sorted(
    {
        hint
        for product_hint in PRODUCT_HINTS
        for hint in product_hint["brand_or_store_hints"]
    }
)

BENEFITS_INTERPRETER_PROMPT = """
Sos un analista de intención comercial para beneficios bancarios.
Tu tarea es interpretar una consulta y devolver únicamente JSON válido.

Reglas:
- No consumas APIs ni inventes promociones.
- Usá solo estas categorías conocidas:
  Supermercados, Gastronomía, Indumentaria, Electrónica, Hogar, Salud y Bienestar,
  Juguetes, Mascotas, Librerías, Viajes, Vehículos, Otros.
- Si detectás planificación de compra, regalo, destinatario u ocasión, reflejalo.
- Si no estás seguro de un campo, devolvelo como null o lista vacía.
- "needs_location" debe ser true.
- "sort_strategy" debe ser "relevance_promotion_distance".
- "brand_or_store_hints" puede incluir marcas o tipos de comercio probables, pero no promociones ni direcciones.

Formato:
{
  "intent": "benefits_search" | "benefits_planning",
  "needs_location": true,
  "commercial_intent": string | null,
  "product_interest": string | null,
  "recipient": string | null,
  "occasion": string | null,
  "category_candidates": string[],
  "store_type_hints": string[],
  "brand_or_store_hints": string[],
  "sort_strategy": "relevance_promotion_distance",
  "audience_hint": "general" | "eminent" | null
}
"""


def extract_benefits_intent(
    query: str | None,
    *,
    llm: ChatOpenAI | None = None,
    reference_date: date | None = None,
) -> dict[str, Any]:
    heuristic = _heuristic_intent(query, reference_date=reference_date)
    llm_payload: dict[str, Any] = {}

    if llm is not None:
        llm_payload = _extract_with_llm(
            query=query or "",
            heuristic=heuristic,
            llm=llm,
        )

    merged = _merge_intent_payloads(heuristic, llm_payload)
    log_step(
        "BENEFITS_INTEL",
        "Intencion de benefits interpretada",
        {
            "query": query or "",
            "intent": merged.get("intent"),
            "commercial_intent": merged.get("commercial_intent"),
            "product_interest": merged.get("product_interest"),
            "recipient": merged.get("recipient"),
            "occasion": merged.get("occasion"),
            "category_candidates": merged.get("category_candidates"),
            "brand_or_store_hints": merged.get("brand_or_store_hints"),
        },
    )
    return merged


def _heuristic_intent(
    query: str | None,
    *,
    reference_date: date | None = None,
) -> dict[str, Any]:
    normalized_query = _normalize_text(query)
    explicit_occasion = detect_explicit_occasion(query, reference_date=reference_date)

    category_candidates: list[str] = []
    for category, aliases in CATEGORY_ALIASES.items():
        if any(_contains_term(normalized_query, alias) for alias in aliases):
            category_candidates.append(category)

    product_interest: str | None = None
    store_type_hints: list[str] = []
    brand_or_store_hints: list[str] = []

    for hint in PRODUCT_HINTS:
        if any(_contains_term(normalized_query, keyword) for keyword in hint["keywords"]):
            if hint["product_interest"] and not product_interest:
                product_interest = str(hint["product_interest"])
            category_candidates.extend(hint["category_candidates"])
            store_type_hints.extend(hint["store_type_hints"])
            brand_or_store_hints.extend(hint["brand_or_store_hints"])

    recipient = _detect_recipient(normalized_query)
    audience_hint = "eminent" if any(_contains_term(normalized_query, term) for term in EMINENT_PATTERNS) else "general"

    commercial_intent = "gift_planning" if _looks_like_gift_planning(normalized_query, recipient) else "direct_discount_search"
    calendar_context = get_active_commercial_context(
        reference_date=reference_date,
        categories=category_candidates or None,
    )
    active_occasion = calendar_context.get("active_occasion")

    if explicit_occasion and explicit_occasion.get("related_categories"):
        category_candidates.extend(explicit_occasion["related_categories"])
    elif (
        not category_candidates
        and active_occasion
        and commercial_intent == "gift_planning"
    ):
        category_candidates.extend(active_occasion.get("related_categories") or [])

    explicit_brand_mentions = _detect_explicit_brand_mentions(
        normalized_query,
        known_hints=KNOWN_STORE_HINTS + brand_or_store_hints,
    )
    brand_or_store_hints.extend(explicit_brand_mentions)

    category_candidates = _dedupe_categories(category_candidates)
    store_type_hints = _dedupe_strings(store_type_hints)
    brand_or_store_hints = _dedupe_strings(brand_or_store_hints)

    occasion = None
    if explicit_occasion:
        occasion = explicit_occasion.get("id")
    elif active_occasion and commercial_intent == "gift_planning":
        occasion = active_occasion.get("id")

    intent = (
        "benefits_planning"
        if commercial_intent == "gift_planning" or product_interest or recipient or occasion
        else "benefits_search"
    )

    return {
        "intent": intent,
        "needs_location": True,
        "commercial_intent": commercial_intent,
        "product_interest": product_interest,
        "recipient": recipient,
        "occasion": occasion,
        "category_candidates": category_candidates,
        "store_type_hints": store_type_hints,
        "brand_or_store_hints": brand_or_store_hints,
        "sort_strategy": "relevance_promotion_distance",
        "audience_hint": audience_hint,
        "active_occasion": active_occasion,
        "explicit_occasion": explicit_occasion,
        "normalized_query": normalized_query,
    }


def _extract_with_llm(
    *,
    query: str,
    heuristic: dict[str, Any],
    llm: ChatOpenAI | None,
) -> dict[str, Any]:
    if llm is None:
        return {}

    try:
        response = llm.invoke(
            [
                ("system", BENEFITS_INTERPRETER_PROMPT),
                (
                    "user",
                    json.dumps(
                        {
                            "query": query,
                            "known_categories": KNOWN_CATEGORIES,
                            "heuristic_guess": {
                                "intent": heuristic.get("intent"),
                                "commercial_intent": heuristic.get("commercial_intent"),
                                "product_interest": heuristic.get("product_interest"),
                                "recipient": heuristic.get("recipient"),
                                "occasion": heuristic.get("occasion"),
                                "category_candidates": heuristic.get("category_candidates"),
                                "store_type_hints": heuristic.get("store_type_hints"),
                                "brand_or_store_hints": heuristic.get("brand_or_store_hints"),
                                "active_occasion": heuristic.get("active_occasion"),
                            },
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
        parsed = _parse_llm_json(getattr(response, "content", ""))
        if not isinstance(parsed, dict):
            return {}
        return parsed
    except Exception as exc:
        log_step("BENEFITS_INTEL", "Fallo el enriquecimiento LLM de intención", {"error": str(exc)})
        return {}


def _merge_intent_payloads(
    heuristic: dict[str, Any],
    llm_payload: dict[str, Any],
) -> dict[str, Any]:
    occasion = heuristic.get("occasion") or _normalize_occasion_id(llm_payload.get("occasion"))
    active_occasion = heuristic.get("active_occasion")
    explicit_occasion = heuristic.get("explicit_occasion")

    category_candidates = _dedupe_categories(
        list(heuristic.get("category_candidates") or [])
        + list(llm_payload.get("category_candidates") or [])
    )
    store_type_hints = _dedupe_strings(
        list(heuristic.get("store_type_hints") or [])
        + list(llm_payload.get("store_type_hints") or [])
    )
    brand_or_store_hints = _dedupe_strings(
        list(heuristic.get("brand_or_store_hints") or [])
        + list(llm_payload.get("brand_or_store_hints") or [])
    )

    product_interest = (
        heuristic.get("product_interest")
        or _safe_string(llm_payload.get("product_interest"))
    )
    recipient = heuristic.get("recipient") or _canonical_recipient(_safe_string(llm_payload.get("recipient")))
    commercial_intent = heuristic.get("commercial_intent") or _safe_string(llm_payload.get("commercial_intent"))

    audience_hint = _safe_string(llm_payload.get("audience_hint")) or heuristic.get("audience_hint") or "general"
    if audience_hint not in {"general", "eminent"}:
        audience_hint = heuristic.get("audience_hint") or "general"

    intent = _safe_string(llm_payload.get("intent")) or heuristic.get("intent") or "benefits_search"
    if commercial_intent == "gift_planning" or product_interest or recipient or occasion:
        intent = "benefits_planning"
    elif intent not in {"benefits_search", "benefits_planning"}:
        intent = "benefits_search"

    return {
        "intent": intent,
        "needs_location": True,
        "commercial_intent": commercial_intent or "direct_discount_search",
        "product_interest": product_interest,
        "recipient": recipient,
        "occasion": occasion,
        "category_candidates": category_candidates,
        "store_type_hints": store_type_hints,
        "brand_or_store_hints": brand_or_store_hints,
        "sort_strategy": "relevance_promotion_distance",
        "audience_hint": audience_hint,
        "active_occasion": active_occasion,
        "explicit_occasion": explicit_occasion,
        "normalized_query": heuristic.get("normalized_query") or _normalize_text(""),
    }


def _detect_recipient(normalized_query: str) -> str | None:
    for canonical, aliases in RECIPIENT_HINTS.items():
        if any(_contains_term(normalized_query, alias) for alias in aliases):
            return canonical
    return None


def _canonical_recipient(value: str | None) -> str | None:
    normalized_value = _normalize_text(value)
    if not normalized_value:
        return None
    for canonical, aliases in RECIPIENT_HINTS.items():
        if normalized_value == canonical or any(_contains_term(normalized_value, alias) for alias in aliases):
            return canonical
    return value.strip() if isinstance(value, str) and value.strip() else None


def _looks_like_gift_planning(normalized_query: str, recipient: str | None) -> bool:
    if recipient:
        return True
    return any(_contains_term(normalized_query, pattern) for pattern in GIFT_PATTERNS)


def _detect_explicit_brand_mentions(
    normalized_query: str,
    known_hints: list[str],
) -> list[str]:
    matches = []
    for hint in known_hints:
        if _contains_term(normalized_query, hint):
            matches.append(hint)
    return matches


def _parse_llm_json(content: Any) -> dict[str, Any] | None:
    if isinstance(content, list):
        text = "\n".join(
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    else:
        text = str(content or "")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _normalize_occasion_id(value: Any) -> str | None:
    normalized_value = _normalize_text(value)
    if not normalized_value:
        return None

    occasion = detect_explicit_occasion(normalized_value)
    if occasion:
        return occasion.get("id")

    return normalized_value.replace(" ", "_")


def _dedupe_categories(categories: list[str]) -> list[str]:
    canonical_by_normalized = {_normalize_text(category): category for category in KNOWN_CATEGORIES}
    deduped: list[str] = []
    seen: set[str] = set()

    for category in categories:
        normalized_category = _normalize_text(category)
        canonical = canonical_by_normalized.get(normalized_category)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(canonical)

    return deduped


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()

    for value in values:
        clean_value = str(value or "").strip()
        normalized_value = _normalize_text(clean_value)
        if not clean_value or not normalized_value or normalized_value in seen:
            continue
        seen.add(normalized_value)
        deduped.append(clean_value)

    return deduped


def _safe_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _contains_term(text: str, term: str) -> bool:
    normalized_term = _normalize_text(term)
    return f" {normalized_term} " in f" {text} "


def _normalize_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    lowered = without_accents.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()
