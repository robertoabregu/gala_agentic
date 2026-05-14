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
        "supermercado",
        "supermercados",
        "superes",
    ),
    "Gastronomía": (
        "gastronomia",
        "gastronomía",
        "restaurantes",
        "restaurante",
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

EMINENT_PATTERNS = (
    "eminent",
    "eminent black",
    "black",
    "exclusivo",
    "exclusivos",
    "seleccion exclusiva",
    "selección exclusiva",
)

QR_PATTERNS = ("qr", "pago qr")
NFC_PATTERNS = ("nfc", "pago nfc", "contactless")
TODAY_PATTERNS = ("hoy", "para hoy")
EVERY_DAY_PATTERNS = ("todos los dias", "todos los días")

IGNORED_QUERY_TOKENS = {
    "a",
    "al",
    "alguno",
    "algun",
    "beneficio",
    "beneficios",
    "black",
    "banco",
    "categoria",
    "categorias",
    "cual",
    "como",
    "con",
    "cuales",
    "cuales",
    "de",
    "del",
    "descuento",
    "descuentos",
    "el",
    "en",
    "eminent",
    "exclusivo",
    "exclusivos",
    "favor",
    "hay",
    "hoy",
    "la",
    "las",
    "los",
    "me",
    "mi",
    "mis",
    "mostrame",
    "mostrar",
    "nfc",
    "oferta",
    "ofertas",
    "para",
    "pago",
    "pedime",
    "por",
    "promo",
    "promocion",
    "promociones",
    "promos",
    "qr",
    "que",
    "quiero",
    "seleccion",
    "si",
    "solo",
    "sus",
    "tengo",
    "tienen",
    "tenemos",
    "tenes",
    "tenés",
    "dia",
    "dias",
    "todos",
    "todas",
    "vigente",
    "vigentes",
}

for aliases in CATEGORY_SYNONYMS.values():
    for alias in aliases:
        normalized_alias = (
            unicodedata.normalize("NFKD", alias)
            .encode("ascii", "ignore")
            .decode("ascii")
            .lower()
        )
        for token in re.split(r"\s+", normalized_alias):
            if token:
                IGNORED_QUERY_TOKENS.add(token)

for day_name in DAYS_ORDER:
    IGNORED_QUERY_TOKENS.add(day_name)


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
        if _contains_term(normalized, category):
            return category

    for category, aliases in CATEGORY_SYNONYMS.items():
        if any(_contains_term(normalized, alias) for alias in aliases):
            return category

    return None


def infer_benefits_filters(text: str | None) -> dict[str, Any]:
    normalized = _normalize_text(text)
    category = resolve_benefit_category(normalized)

    return {
        "category": category,
        "only_eminent": any(_contains_term(normalized, pattern) for pattern in EMINENT_PATTERNS),
        "only_qr": any(_contains_term(normalized, pattern) for pattern in QR_PATTERNS),
        "only_nfc": any(_contains_term(normalized, pattern) for pattern in NFC_PATTERNS),
        "today_only": any(_contains_term(normalized, pattern) for pattern in TODAY_PATTERNS),
        "every_day_only": any(_contains_term(normalized, pattern) for pattern in EVERY_DAY_PATTERNS),
        "search_terms": _extract_search_terms(normalized),
    }


def search_benefits(
    category: str | None = None,
    query: str | None = None,
    only_eminent: bool = False,
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
    day_filter = _today_day_name() if today_only else None

    results: list[dict[str, Any]] = []

    for benefit in _iter_benefits():
        if canonical_category and benefit["categoria"] != canonical_category:
            continue

        if only_eminent and not benefit.get("exclusivoEminent"):
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


def _extract_search_terms(text: str) -> list[str]:
    return [
        token
        for token in _normalize_text(text).split()
        if len(token) >= 2 and token not in IGNORED_QUERY_TOKENS
    ]


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
