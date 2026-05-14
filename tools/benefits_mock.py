from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
MOCK_BENEFITS_PATH = BASE_DIR / "data" / "mock_benefits.json"

DAYS_ORDER = [
    "lunes",
    "martes",
    "miercoles",
    "jueves",
    "viernes",
    "sabado",
    "domingo",
]

CATEGORY_SYNONYMS = {
    "Supermercados": (
        "super",
        "súper",
        "supermercado",
        "supermercados",
        "mercado",
        "superes",
        "compras del super",
        "promos del super",
        "beneficios del super",
    ),
    "Gastronomía": (
        "gastronomia",
        "gastronomía",
        "restaurante",
        "restaurantes",
        "comida",
        "cafe",
        "café",
    ),
    "Indumentaria": (
        "ropa",
        "indumentaria",
        "zapatillas",
    ),
    "Electrónica": (
        "electronica",
        "electrónica",
        "celulares",
        "celular",
        "tecnologia",
        "tecnología",
    ),
    "Hogar": (
        "hogar",
        "casa",
        "muebles",
        "pintureria",
        "pinturería",
    ),
}

ONLY_EMINENT_PATTERNS = (
    "eminent",
    "eminent black",
    "por ser eminent",
    "para eminent",
    "para eminent black",
    "beneficios eminent",
    "beneficios para eminent",
    "beneficios por ser eminent",
    "solo eminent",
    "solo beneficios eminent",
    "solo exclusivos",
    "solo exclusivas",
    "solo exclusivo",
    "solo exclusiva",
    "beneficios exclusivos",
    "beneficios exclusivas",
    "promo exclusiva",
    "promos exclusivas",
    "promos exclusivos",
)

EXCLUDE_EMINENT_PATTERNS = (
    "no eminent",
    "no sean eminent",
    "no sean para eminent",
    "no exclusivo",
    "no exclusiva",
    "no exclusivos",
    "no exclusivas",
    "que no sean exclusivos",
    "que no sean exclusivas",
    "que no sean para eminent",
    "que no sean eminent",
    "comun",
    "comunes",
)

EMINENT_TERMS = (
    "eminent",
    "eminent black",
)

EXCLUSIVE_TERMS = (
    "exclusivo",
    "exclusiva",
    "exclusivos",
    "exclusivas",
)

QR_PATTERNS = ("qr", "pago qr")
NFC_PATTERNS = ("nfc", "pago nfc", "contactless")
TODAY_PATTERNS = ("hoy", "para hoy")
EVERY_DAY_PATTERNS = ("todos los dias", "todos los días")

IGNORED_QUERY_TOKENS = {
    "a",
    "al",
    "algo",
    "algun",
    "alguno",
    "beneficio",
    "beneficios",
    "black",
    "banco",
    "cafe",
    "casa",
    "categoria",
    "categorias",
    "comida",
    "comun",
    "comunes",
    "compra",
    "compras",
    "con",
    "cual",
    "cuales",
    "de",
    "del",
    "descuento",
    "descuentos",
    "dia",
    "dias",
    "el",
    "eminent",
    "en",
    "esos",
    "esas",
    "exclusiva",
    "exclusivas",
    "exclusivo",
    "exclusivos",
    "favor",
    "hay",
    "hogar",
    "hoy",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "mercado",
    "mi",
    "mis",
    "mostrame",
    "mostrar",
    "muebles",
    "nfc",
    "no",
    "oferta",
    "ofertas",
    "pago",
    "para",
    "pedime",
    "por",
    "promo",
    "promocion",
    "promociones",
    "promos",
    "qr",
    "que",
    "quiero",
    "sea",
    "sean",
    "seleccion",
    "ser",
    "si",
    "solo",
    "sus",
    "tengo",
    "tenemos",
    "tenes",
    "tenés",
    "tienen",
    "todos",
    "todas",
    "un",
    "una",
    "vigente",
    "vigentes",
    "y",
    "yo",
}


def _normalize_text(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    lowered = without_accents.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


for aliases in CATEGORY_SYNONYMS.values():
    for alias in aliases:
        for token in _normalize_text(alias).split():
            if token:
                IGNORED_QUERY_TOKENS.add(token)

for day_name in DAYS_ORDER:
    IGNORED_QUERY_TOKENS.add(day_name)


def _contains_term(text: str, term: str) -> bool:
    normalized_text = f" {_normalize_text(text)} "
    normalized_term = f" {_normalize_text(term)} "
    return normalized_term in normalized_text


@lru_cache(maxsize=1)
def load_mock_benefits() -> dict[str, Any]:
    with MOCK_BENEFITS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("mock_benefits.json debe contener un objeto JSON.")

    return data


def get_benefits_segment() -> str:
    return str(load_mock_benefits().get("segmento") or "").strip()


def list_benefit_categories() -> list[str]:
    categories = load_mock_benefits().get("categorias") or []
    return [
        str(category.get("categoria") or "").strip()
        for category in categories
        if isinstance(category, dict) and str(category.get("categoria") or "").strip()
    ]


def resolve_benefit_category(text: str | None) -> str | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None

    for category in list_benefit_categories():
        if _text_matches_category(normalized, category):
            return category

    for category, aliases in CATEGORY_SYNONYMS.items():
        if any(_text_matches_alias(normalized, alias) for alias in aliases):
            return category

    return None


def infer_benefits_filters(text: str | None) -> dict[str, Any]:
    normalized = _normalize_text(text)
    category = resolve_benefit_category(normalized)
    exclude_eminent = _has_exclude_eminent_intent(normalized)
    only_eminent = _has_only_eminent_intent(normalized) and not exclude_eminent

    return {
        "category": category,
        "only_eminent": only_eminent,
        "exclude_eminent": exclude_eminent,
        "only_qr": any(_contains_term(normalized, pattern) for pattern in QR_PATTERNS),
        "only_nfc": any(_contains_term(normalized, pattern) for pattern in NFC_PATTERNS),
        "today_only": any(_contains_term(normalized, pattern) for pattern in TODAY_PATTERNS),
        "every_day_only": any(_contains_term(normalized, pattern) for pattern in EVERY_DAY_PATTERNS),
        "search_terms": _extract_search_terms(normalized, category=category),
    }


def search_benefits(
    category: str | None = None,
    query: str | None = None,
    only_eminent: bool = False,
    exclude_eminent: bool = False,
    limit: int = 5,
    *,
    only_qr: bool = False,
    only_nfc: bool = False,
    today_only: bool = False,
    every_day_only: bool = False,
) -> list[dict[str, Any]]:
    inferred_filters = infer_benefits_filters(query)
    canonical_category = resolve_benefit_category(category) or inferred_filters["category"]
    search_terms = inferred_filters["search_terms"]
    should_exclude_eminent = exclude_eminent or inferred_filters["exclude_eminent"]
    should_only_eminent = only_eminent and not should_exclude_eminent
    day_filter = _today_day_name() if today_only else None

    results: list[dict[str, Any]] = []

    for benefit in _iter_benefits():
        if canonical_category and benefit["categoria"] != canonical_category:
            continue

        if should_exclude_eminent and benefit.get("exclusivoEminent"):
            continue

        if should_only_eminent and not benefit.get("exclusivoEminent"):
            continue

        if only_qr and not benefit.get("pagoQR"):
            continue

        if only_nfc and not benefit.get("pagoNFC"):
            continue

        if every_day_only and not _is_every_day(benefit.get("dias")):
            continue

        if day_filter and not _matches_day(benefit.get("dias"), day_filter):
            continue

        if search_terms and not _matches_query_terms(benefit, search_terms):
            continue

        results.append(benefit)

    return results[: max(1, limit)]


def _iter_benefits() -> list[dict[str, Any]]:
    data = load_mock_benefits()
    categories = data.get("categorias") or []
    segment = get_benefits_segment()
    items: list[dict[str, Any]] = []

    for category_entry in categories:
        if not isinstance(category_entry, dict):
            continue

        category_name = str(category_entry.get("categoria") or "").strip()
        benefits = category_entry.get("beneficios") or []

        if not category_name or not isinstance(benefits, list):
            continue

        for benefit in benefits:
            if not isinstance(benefit, dict):
                continue

            items.append(
                {
                    **benefit,
                    "categoria": category_name,
                    "segmento": segment,
                }
            )

    return items


def _extract_search_terms(text: str, *, category: str | None = None) -> list[str]:
    cleaned_text = _strip_category_terms(text, category)

    return [
        token
        for token in cleaned_text.split()
        if len(token) >= 3
        and token not in IGNORED_QUERY_TOKENS
        and not _token_is_category_related(token, category)
    ]


def _strip_category_terms(text: str, category: str | None) -> str:
    cleaned_text = f" {_normalize_text(text)} "
    aliases: list[str] = []

    if category:
        aliases.append(category)
        aliases.extend(CATEGORY_SYNONYMS.get(category, ()))

    for alias in aliases:
        normalized_alias = _normalize_text(alias)
        if not normalized_alias:
            continue
        cleaned_text = cleaned_text.replace(f" {normalized_alias} ", " ")

    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.strip()


def _text_matches_category(text: str, category: str) -> bool:
    if _contains_term(text, category):
        return True

    return _text_matches_alias(text, category)


def _text_matches_alias(text: str, alias: str) -> bool:
    if _contains_term(text, alias):
        return True

    normalized_text = _normalize_text(text)
    normalized_alias = _normalize_text(alias)
    if " " in normalized_alias:
        return False

    alias_tokens = normalized_alias.split()
    text_tokens = normalized_text.split()

    for alias_token in alias_tokens:
        if len(alias_token) < 5 or alias_token in IGNORED_QUERY_TOKENS:
            continue

        for text_token in text_tokens:
            if len(text_token) < 5 or text_token in IGNORED_QUERY_TOKENS:
                continue

            if alias_token.startswith(text_token) or text_token.startswith(alias_token):
                return True

    return False


def _token_is_category_related(token: str, category: str | None) -> bool:
    if not category:
        return False

    normalized_token = _normalize_text(token)
    if len(normalized_token) < 4:
        return False

    aliases = [category, *CATEGORY_SYNONYMS.get(category, ())]

    for alias in aliases:
        for alias_token in _normalize_text(alias).split():
            if len(alias_token) < 4:
                continue

            if alias_token.startswith(normalized_token) or normalized_token.startswith(alias_token):
                return True

    return False


def _has_exclude_eminent_intent(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    if any(_contains_term(normalized, pattern) for pattern in EXCLUDE_EMINENT_PATTERNS):
        return True

    negative_patterns = (
        r"\bno\b.*\beminent\b",
        r"\bno\b.*\bexclusiv\w*\b",
        r"\bque no sean\b.*\beminent\b",
        r"\bque no sean\b.*\bexclusiv\w*\b",
    )

    return any(re.search(pattern, normalized) for pattern in negative_patterns)


def _has_only_eminent_intent(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    if any(_contains_term(normalized, pattern) for pattern in ONLY_EMINENT_PATTERNS):
        return True

    has_eminent_term = any(_contains_term(normalized, term) for term in EMINENT_TERMS)
    has_exclusive_term = any(_contains_term(normalized, term) for term in EXCLUSIVE_TERMS)

    return has_eminent_term or has_exclusive_term


def _matches_query_terms(benefit: dict[str, Any], search_terms: list[str]) -> bool:
    searchable_text = _normalize_text(
        " ".join(
            [
                str(benefit.get("categoria") or ""),
                str(benefit.get("comercio") or ""),
                str(benefit.get("beneficio") or ""),
                " ".join(benefit.get("mediosDePago") or []),
            ]
        )
    )
    return all(term in searchable_text for term in search_terms)


def _today_day_name() -> str:
    weekday_index = datetime.now().weekday()
    return DAYS_ORDER[weekday_index]


def _is_every_day(days_text: str | None) -> bool:
    return _normalize_text(days_text) == "todos los dias"


def _matches_day(days_text: str | None, day_name: str) -> bool:
    normalized_days = _normalize_text(days_text)
    normalized_day = _normalize_text(day_name)

    if not normalized_days or not normalized_day:
        return False

    if normalized_days == "todos los dias":
        return True

    if _contains_term(normalized_days, normalized_day):
        return True

    if " a " in normalized_days:
        start_day, end_day = [part.strip() for part in normalized_days.split(" a ", maxsplit=1)]
        range_days = _expand_day_range(start_day, end_day)
        return normalized_day in range_days

    return False


def _expand_day_range(start_day: str, end_day: str) -> list[str]:
    if start_day not in DAYS_ORDER or end_day not in DAYS_ORDER:
        return []

    start_index = DAYS_ORDER.index(start_day)
    end_index = DAYS_ORDER.index(end_day)

    if start_index <= end_index:
        return DAYS_ORDER[start_index : end_index + 1]

    return DAYS_ORDER[start_index:] + DAYS_ORDER[: end_index + 1]
