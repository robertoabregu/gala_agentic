from __future__ import annotations

import os
import re
import time
import unicodedata
from datetime import datetime
from functools import lru_cache
from typing import Any

import requests

from observability.logger import log_step
from tools import benefits_mock as mock_benefits


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


BENEFITS_API_BASE_URL = os.getenv(
    "BENEFITS_API_BASE_URL",
    "https://loyalty.bff.bancogalicia.com.ar/api/portal/personalizacion/v1",
).rstrip("/")
BENEFITS_CAROUSEL_ID = os.getenv("BENEFITS_CAROUSEL_ID", "152").strip() or "152"
BENEFITS_PAGE_SIZE = max(1, _get_int_env("BENEFITS_PAGE_SIZE", 50))
BENEFITS_REQUEST_TIMEOUT = max(1.0, _get_float_env("BENEFITS_REQUEST_TIMEOUT", 8))
BENEFITS_USE_REAL_API = _get_bool_env("BENEFITS_USE_REAL_API", True)
BENEFITS_CACHE_TTL_SECONDS = max(60, _get_int_env("BENEFITS_CACHE_TTL_SECONDS", 600))
BENEFITS_SEGMENT = "Eminent Black"

REQUEST_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Id_channel": "onlinebanking",
    "Id_canal": "Quiero",
    "Origin": "https://beneficios.galicia.ar",
    "Referer": "https://beneficios.galicia.ar/",
    "User-Agent": "Mozilla/5.0",
}

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
    "Transportes": (
        "transporte",
        "transportes",
        "viaje",
        "viajes",
        "turismo",
        "pasajes",
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

BENEFITS_STOPWORDS = {
    "alguna",
    "algunas",
    "alguno",
    "algunos",
    "beneficio",
    "beneficios",
    "busca",
    "buscar",
    "categoria",
    "categoría",
    "consulta",
    "consultar",
    "con",
    "contame",
    "dame",
    "de",
    "decime",
    "del",
    "descuento",
    "descuentos",
    "disponible",
    "disponibles",
    "el",
    "en",
    "hay",
    "la",
    "las",
    "los",
    "me",
    "mi",
    "mis",
    "mostrar",
    "mostrame",
    "mostrames",
    "mostrarlos",
    "oferta",
    "ofertas",
    "para",
    "por",
    "promo",
    "promocion",
    "promociones",
    "promos",
    "promoción",
    "que",
    "qué",
    "quiero",
    "quisiera",
    "rubro",
    "sin",
    "sobre",
    "tenes",
    "tenés",
    "tengo",
    "tienen",
    "un",
    "una",
    "unos",
    "unas",
    "ver",
}

GENERIC_QUERY_TOKENS = {
    "a",
    "al",
    "algo",
    "algun",
    "alguna",
    "algunas",
    "alguno",
    "algunos",
    "banco",
    "black",
    "beneficio",
    "beneficios",
    "busca",
    "buscar",
    "categoria",
    "categorias",
    "comun",
    "comunes",
    "con",
    "consulta",
    "consultar",
    "contame",
    "cual",
    "cuales",
    "dame",
    "de",
    "decime",
    "del",
    "descuento",
    "descuentos",
    "dia",
    "dias",
    "disponible",
    "disponibles",
    "el",
    "eminent",
    "en",
    "esas",
    "esos",
    "exclusiva",
    "exclusivas",
    "exclusivo",
    "exclusivos",
    "favor",
    "hay",
    "hoy",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "mi",
    "mis",
    "mostrar",
    "mostrame",
    "mostrames",
    "mostrarlos",
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
    "quisiera",
    "rubro",
    "sea",
    "sean",
    "seleccion",
    "ser",
    "si",
    "sin",
    "solo",
    "sobre",
    "sus",
    "tenemos",
    "tenes",
    "tengo",
    "tenés",
    "tienen",
    "todos",
    "todas",
    "un",
    "una",
    "unos",
    "unas",
    "ver",
    "vigente",
    "vigentes",
    "y",
    "yo",
}

_CACHE_MISS = object()
_CATEGORIES_CACHE: dict[str, Any] = {"expires_at": 0.0, "value": [], "source": "empty"}
_PROMOTIONS_CACHE: dict[str, Any] = {"expires_at": 0.0, "value": [], "source": "empty"}


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


NORMALIZED_BENEFITS_STOPWORDS = {
    _normalize_text(word)
    for word in BENEFITS_STOPWORDS
    if _normalize_text(word)
}

NORMALIZED_GENERIC_QUERY_TOKENS = {
    _normalize_text(word)
    for word in GENERIC_QUERY_TOKENS
    if _normalize_text(word)
}

for aliases in CATEGORY_SYNONYMS.values():
    for alias in aliases:
        normalized_alias = _normalize_text(alias)
        if not normalized_alias:
            continue
        NORMALIZED_GENERIC_QUERY_TOKENS.add(normalized_alias)
        for token in normalized_alias.split():
            NORMALIZED_GENERIC_QUERY_TOKENS.add(token)

for day_name in DAYS_ORDER:
    NORMALIZED_GENERIC_QUERY_TOKENS.add(day_name)


def _contains_term(text: str, term: str) -> bool:
    normalized_text = f" {_normalize_text(text)} "
    normalized_term = f" {_normalize_text(term)} "
    return normalized_term in normalized_text


def _read_cache(cache_entry: dict[str, Any]) -> Any:
    if time.time() < float(cache_entry.get("expires_at", 0.0)):
        return cache_entry.get("value")
    return _CACHE_MISS


def _write_cache(cache_entry: dict[str, Any], value: Any, *, source: str) -> None:
    cache_entry["expires_at"] = time.time() + BENEFITS_CACHE_TTL_SECONDS
    cache_entry["value"] = value
    cache_entry["source"] = source


def _clear_benefits_cache() -> None:
    _CATEGORIES_CACHE.update({"expires_at": 0.0, "value": [], "source": "empty"})
    _PROMOTIONS_CACHE.update({"expires_at": 0.0, "value": [], "source": "empty"})
    _merchant_index.cache_clear()


def get_benefits_segment() -> str:
    return BENEFITS_SEGMENT


def list_benefit_categories() -> list[str]:
    cached = _read_cache(_CATEGORIES_CACHE)
    if cached is not _CACHE_MISS:
        return list(cached)

    if not BENEFITS_USE_REAL_API:
        categories = mock_benefits.list_benefit_categories()
        _write_cache(_CATEGORIES_CACHE, categories, source="fallback_config")
        return list(categories)

    try:
        categories = _fetch_live_categories()
        if not categories:
            raise ValueError("La API de categorias no devolvio elementos.")

        _write_cache(_CATEGORIES_CACHE, categories, source="api")
        log_step("BENEFITS_API", "Categorias actualizadas desde la API", {"results": len(categories)})
        return list(categories)
    except Exception as exc:
        log_step(
            "BENEFITS_API",
            "Fallback al mock para categorias de beneficios",
            {"error": str(exc)},
        )
        categories = mock_benefits.list_benefit_categories()
        _write_cache(_CATEGORIES_CACHE, categories, source="fallback_error")
        return list(categories)


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
    merchant_names = _detect_merchants(normalized, category=category)
    search_terms = _build_search_terms(
        normalized,
        category=category,
        merchant_names=merchant_names,
    )
    cleaned_query = " ".join(search_terms).strip()

    return {
        "category": category,
        "only_eminent": only_eminent,
        "exclude_eminent": exclude_eminent,
        "only_qr": any(_contains_term(normalized, pattern) for pattern in QR_PATTERNS),
        "only_nfc": any(_contains_term(normalized, pattern) for pattern in NFC_PATTERNS),
        "today_only": any(_contains_term(normalized, pattern) for pattern in TODAY_PATTERNS),
        "every_day_only": any(_contains_term(normalized, pattern) for pattern in EVERY_DAY_PATTERNS),
        "raw_query": text or "",
        "cleaned_query": cleaned_query,
        "search_terms": search_terms,
        "merchant_names": merchant_names,
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
    search_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    inferred_filters = infer_benefits_filters(query)
    canonical_category = resolve_benefit_category(category) or inferred_filters["category"]
    normalized_search_terms = [
        _normalize_text(term)
        for term in (search_terms if search_terms is not None else inferred_filters["search_terms"])
        if _normalize_text(term)
    ]
    should_exclude_eminent = exclude_eminent or inferred_filters["exclude_eminent"]
    should_only_eminent = only_eminent and not should_exclude_eminent
    day_filter = _today_day_name() if today_only else None

    results: list[dict[str, Any]] = []

    for benefit in _iter_benefits():
        if canonical_category and not _benefit_matches_category(benefit.get("categoria"), canonical_category):
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

        if normalized_search_terms and not _matches_query_terms(benefit, normalized_search_terms):
            continue

        results.append(benefit)

    return results[: max(1, limit)]


def _iter_benefits() -> list[dict[str, Any]]:
    cached = _read_cache(_PROMOTIONS_CACHE)
    if cached is not _CACHE_MISS:
        return list(cached)

    if not BENEFITS_USE_REAL_API:
        benefits = _load_mock_promotions()
        _write_cache(_PROMOTIONS_CACHE, benefits, source="fallback_config")
        _merchant_index.cache_clear()
        return list(benefits)

    try:
        benefits = _fetch_live_promotions()
        _write_cache(_PROMOTIONS_CACHE, benefits, source="api")
        _merchant_index.cache_clear()
        log_step("BENEFITS_API", "Promociones actualizadas desde la API", {"results": len(benefits)})
        return list(benefits)
    except Exception as exc:
        log_step(
            "BENEFITS_API",
            "Fallback al mock para promociones de beneficios",
            {"error": str(exc)},
        )
        benefits = _load_mock_promotions()
        _write_cache(_PROMOTIONS_CACHE, benefits, source="fallback_error")
        _merchant_index.cache_clear()
        return list(benefits)


def _fetch_live_categories() -> list[str]:
    payload = _fetch_json(
        "categorias",
        params={
            "idAudiencia": 1,
            "SubCategoria": "false",
            "Visibles": "true",
        },
    )
    items = ((payload.get("data") or {}).get("list")) or []
    if not isinstance(items, list):
        raise ValueError("La respuesta de categorias no contiene una lista valida.")

    categories: list[str] = []
    seen: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        description = str(item.get("descripcion") or "").strip()
        if not description or description in seen:
            continue

        seen.add(description)
        categories.append(description)

    return categories


def _fetch_live_promotions() -> list[dict[str, Any]]:
    page = 1
    total_size: int | None = None
    promotions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    while True:
        payload = _fetch_json(
            f"promociones/list/carrusel/{BENEFITS_CAROUSEL_ID}",
            params={
                "page": page,
                "pageSize": BENEFITS_PAGE_SIZE,
                "cardEspecial": "true",
            },
        )
        promotions_block = ((payload.get("data") or {}).get("promociones")) or {}
        page_items = promotions_block.get("list") or []
        if not isinstance(page_items, list):
            raise ValueError("La respuesta de promociones no contiene una lista valida.")

        block_total_size = _safe_int(promotions_block.get("totalSize"))
        if total_size is None and block_total_size is not None:
            total_size = block_total_size

        added_this_page = 0

        for promo in page_items:
            if not isinstance(promo, dict):
                continue

            benefit = _normalize_promotion(promo)
            dedupe_key = str(benefit.get("id") or "")
            if not dedupe_key:
                dedupe_key = "|".join(
                    [
                        str(benefit.get("comercio") or ""),
                        str(benefit.get("beneficio") or ""),
                        str(benefit.get("categoria") or ""),
                        str(page),
                    ]
                )

            if dedupe_key in seen_ids:
                continue

            seen_ids.add(dedupe_key)
            promotions.append(benefit)
            added_this_page += 1

        if total_size is not None and len(promotions) >= total_size:
            break

        if not page_items or added_this_page == 0:
            break

        if len(page_items) < BENEFITS_PAGE_SIZE and total_size is None:
            break

        page += 1
        if page > 100:
            break

    return promotions


def _fetch_json(path: str, *, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{BENEFITS_API_BASE_URL}/{path.lstrip('/')}"
    response = requests.get(
        url,
        headers=REQUEST_HEADERS,
        params=params,
        timeout=BENEFITS_REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("La API de beneficios devolvio un payload invalido.")

    if payload.get("errors"):
        raise ValueError(f"La API de beneficios devolvio errores: {payload['errors']}")

    return payload


def _normalize_promotion(promo: dict[str, Any]) -> dict[str, Any]:
    modelo_atencion = promo.get("modeloAtencion") or {}
    if not isinstance(modelo_atencion, dict):
        modelo_atencion = {}

    commerce = str(promo.get("titulo") or "").strip() or "Beneficio"
    benefit_text = str(promo.get("promocion") or "").strip() or "Beneficio disponible"
    category = str(promo.get("subtitulo") or "").strip() or "Otros"
    end_date = promo.get("fechaHasta")

    return {
        "id": promo.get("id"),
        "comercio": commerce,
        "beneficio": benefit_text,
        "categoria": category,
        "dias": str(promo.get("leyendaDiasAplicacion") or "").strip(),
        "mediosDePago": _normalize_payment_methods(promo.get("mediosDePago") or []),
        "exclusivoEminent": bool(promo.get("eminent")) or bool(modelo_atencion.get("exclusivo")),
        "pagoQR": bool(promo.get("pagoQR")),
        "pagoNFC": bool(promo.get("pagoNFC")),
        "proximamente": bool(promo.get("proximamente")),
        "fechaHasta": end_date,
        "vigenciaHasta": end_date,
        "imagen": promo.get("imagen"),
        "tipoPromocion": promo.get("tipoPromocion"),
        "segmento": BENEFITS_SEGMENT,
    }


def _normalize_payment_methods(payment_methods: list[Any]) -> list[str]:
    normalized_methods: list[str] = []
    seen: set[str] = set()

    for method in payment_methods:
        if not isinstance(method, dict):
            continue

        normalized_type = _normalize_payment_type(method.get("tipoTarjeta"))
        if not normalized_type or normalized_type in seen:
            continue

        seen.add(normalized_type)
        normalized_methods.append(normalized_type)

    return normalized_methods


def _normalize_existing_payment_methods(payment_methods: list[Any]) -> list[str]:
    normalized_methods: list[str] = []
    seen: set[str] = set()

    for method in payment_methods:
        normalized_type = _normalize_payment_type(method)
        if not normalized_type or normalized_type in seen:
            continue

        seen.add(normalized_type)
        normalized_methods.append(normalized_type)

    return normalized_methods


def _normalize_payment_type(value: Any) -> str | None:
    raw_value = str(value or "").strip()
    normalized_value = _normalize_text(raw_value)

    if not normalized_value:
        return None

    if normalized_value == "credito":
        return "Crédito"

    if normalized_value == "debito":
        return "Débito"

    return raw_value.title()


def _load_mock_promotions() -> list[dict[str, Any]]:
    data = mock_benefits.load_mock_benefits()
    categories = data.get("categorias") or []
    benefits: list[dict[str, Any]] = []

    for category_entry in categories:
        if not isinstance(category_entry, dict):
            continue

        category_name = str(category_entry.get("categoria") or "").strip() or "Otros"
        category_benefits = category_entry.get("beneficios") or []

        if not isinstance(category_benefits, list):
            continue

        for benefit in category_benefits:
            if not isinstance(benefit, dict):
                continue

            end_date = benefit.get("fechaHasta") or benefit.get("vigenciaHasta")

            benefits.append(
                {
                    "id": benefit.get("id"),
                    "comercio": str(benefit.get("comercio") or "").strip() or "Beneficio",
                    "beneficio": str(benefit.get("beneficio") or "").strip() or "Beneficio disponible",
                    "categoria": category_name,
                    "dias": str(benefit.get("dias") or "").strip(),
                    "mediosDePago": _normalize_existing_payment_methods(
                        benefit.get("mediosDePago") or []
                    ),
                    "exclusivoEminent": bool(benefit.get("exclusivoEminent")),
                    "pagoQR": bool(benefit.get("pagoQR")),
                    "pagoNFC": bool(benefit.get("pagoNFC")),
                    "proximamente": bool(benefit.get("proximamente")),
                    "fechaHasta": end_date,
                    "vigenciaHasta": end_date,
                    "imagen": benefit.get("imagen"),
                    "tipoPromocion": benefit.get("tipoPromocion"),
                    "segmento": get_benefits_segment(),
                }
            )

    return benefits


@lru_cache(maxsize=1)
def _merchant_index() -> list[dict[str, Any]]:
    merchants: dict[str, dict[str, Any]] = {}

    for benefit in _iter_benefits():
        merchant_name = str(benefit.get("comercio") or "").strip()
        normalized_name = _normalize_text(merchant_name)
        if not merchant_name or not normalized_name:
            continue

        entry = merchants.setdefault(
            normalized_name,
            {
                "name": merchant_name,
                "normalized_name": normalized_name,
                "categories": set(),
                "aliases": set(),
            },
        )
        entry["categories"].add(str(benefit.get("categoria") or "").strip())
        entry["aliases"].add(normalized_name)

        for token in normalized_name.split():
            if len(token) >= 5 and token not in NORMALIZED_GENERIC_QUERY_TOKENS:
                entry["aliases"].add(token)

    indexed_merchants: list[dict[str, Any]] = []
    for entry in merchants.values():
        indexed_merchants.append(
            {
                "name": entry["name"],
                "normalized_name": entry["normalized_name"],
                "categories": tuple(sorted(entry["categories"])),
                "aliases": tuple(sorted(entry["aliases"], key=lambda alias: (-len(alias), alias))),
            }
        )

    return sorted(
        indexed_merchants,
        key=lambda item: (-len(item["normalized_name"]), item["normalized_name"]),
    )


def _detect_merchants(text: str, *, category: str | None = None) -> list[str]:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return []

    matches: list[str] = []

    for merchant in _merchant_index():
        if category and not any(
            _benefit_matches_category(merchant_category, category)
            for merchant_category in merchant["categories"]
        ):
            continue

        if any(_contains_term(normalized_text, alias) for alias in merchant["aliases"]):
            matches.append(merchant["name"])

    if not matches:
        return []

    return [matches[0]]


def _build_search_terms(
    text: str,
    *,
    category: str | None,
    merchant_names: list[str],
) -> list[str]:
    if merchant_names:
        primary_merchant = merchant_names[0]
        return [_normalize_text(primary_merchant)]

    if category:
        return []

    return _extract_free_text_terms(text)


def _extract_free_text_terms(text: str) -> list[str]:
    cleaned_text = _normalize_text(text)
    return [
        token
        for token in cleaned_text.split()
        if len(token) >= 3 and token not in NORMALIZED_GENERIC_QUERY_TOKENS
    ]


def _benefit_matches_category(benefit_category: Any, requested_category: str | None) -> bool:
    if not requested_category:
        return True

    benefit_text = str(benefit_category or "").strip()
    if not benefit_text:
        return False

    normalized_benefit = _normalize_text(benefit_text)
    normalized_requested = _normalize_text(requested_category)
    if normalized_benefit == normalized_requested:
        return True

    benefit_canonical = resolve_benefit_category(benefit_text) or benefit_text
    requested_canonical = resolve_benefit_category(requested_category) or requested_category
    if _normalize_text(benefit_canonical) == _normalize_text(requested_canonical):
        return True

    if _text_matches_category(benefit_text, requested_category):
        return True

    aliases = CATEGORY_SYNONYMS.get(requested_canonical, ())
    return any(_text_matches_alias(benefit_text, alias) for alias in aliases)


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
        if len(alias_token) < 5 or alias_token in NORMALIZED_GENERIC_QUERY_TOKENS:
            continue

        for text_token in text_tokens:
            if len(text_token) < 5 or text_token in NORMALIZED_GENERIC_QUERY_TOKENS:
                continue

            if alias_token.startswith(text_token) or text_token.startswith(alias_token):
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


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
