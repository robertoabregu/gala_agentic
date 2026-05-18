from __future__ import annotations

import os
import re
import time
import unicodedata
from datetime import datetime
from functools import lru_cache
from typing import Any, Iterable
from urllib.parse import quote

import requests

from observability.logger import log_step


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
BENEFITS_REQUEST_TIMEOUT = max(1.0, _get_float_env("BENEFITS_REQUEST_TIMEOUT", 8))
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
        "superes",
        "supermercado",
        "supermercados",
        "changomas",
        "carrefour",
        "disco",
        "jumbo",
        "vea",
    ),
    "Gastronomía": (
        "gastronomia",
        "comida",
        "comer",
        "delivery",
        "pedir comida",
        "restaurante",
        "restaurantes",
        "cafe",
        "cafeteria",
        "helado",
        "hamburguesa",
        "sushi",
    ),
    "Indumentaria": (
        "ropa",
        "indumentaria",
        "zapatilla",
        "zapatillas",
        "zapas",
        "calzado",
        "moda",
        "vestimenta",
    ),
    "Electrónica": (
        "electronica",
        "electro",
        "electrodomestico",
        "electrodomesticos",
        "tecnologia",
        "celular",
        "celulares",
        "notebook",
        "tv",
        "televisor",
        "heladera",
        "lavarropas",
    ),
    "Hogar": (
        "hogar",
        "casa",
        "muebles",
        "deco",
        "decoracion",
        "bazar",
        "colchon",
        "colchones",
    ),
    "Vehículos": (
        "vehiculo",
        "vehiculos",
        "auto",
        "autos",
        "moto",
        "motos",
        "neumatico",
        "neumaticos",
    ),
    "Salud y Bienestar": (
        "salud",
        "bienestar",
        "farmacia",
        "farmacias",
        "perfumeria",
        "gimnasio",
    ),
    "Viajes": (
        "viaje",
        "viajes",
        "turismo",
        "hotel",
        "hoteles",
        "vuelo",
        "vuelos",
        "pasaje",
        "pasajes",
    ),
    "Entretenimiento": (
        "entretenimiento",
        "cine",
        "teatro",
        "recital",
        "show",
        "streaming",
    ),
    "Librerías": (
        "libreria",
        "librerias",
        "libro",
        "libros",
        "utiles",
        "escolares",
    ),
    "Shopping": (
        "shopping",
        "shoppings",
        "mall",
        "outlet",
    ),
    "Mascotas": (
        "mascota",
        "mascotas",
        "pet",
        "veterinaria",
        "perro",
        "gato",
    ),
    "Juguetes": (
        "juguete",
        "juguetes",
        "juego",
        "juegos",
    ),
    "Transportes": (
        "transporte",
        "transportes",
        "sube",
        "taxi",
        "remis",
        "colectivo",
    ),
}

ONLY_EMINENT_PATTERNS = (
    "eminent",
    "eminent black",
    "por ser eminent",
    "soy eminent",
    "cliente eminent",
    "para eminent",
    "para eminent black",
    "beneficios eminent",
    "beneficios para eminent",
    "beneficios por ser eminent",
    "exclusivo eminent",
    "exclusivos eminent",
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

QR_PATTERNS = ("qr", "pago qr", "pagos con qr", "promos con qr")
NFC_PATTERNS = (
    "nfc",
    "pago nfc",
    "pagos con nfc",
    "promos con nfc",
    "contactless",
    "contact less",
    "sin contacto",
    "pago sin contacto",
)
TODAY_PATTERNS = ("hoy", "para hoy")
EVERY_DAY_PATTERNS = ("todos los dias", "todos los días")
INTEREST_FREE_PATTERNS = ("sin interes", "sin interés")
INSTALLMENTS_PATTERNS = (
    "cuota",
    "cuotas",
    "pagar en cuotas",
    "pagar cuotas",
    "sacar cuotas",
)

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
    "categorias",
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
    "este",
    "favor",
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
    "necesito",
    "nueva",
    "nuevas",
    "nuevo",
    "nuevos",
    "oferta",
    "ofertas",
    "para",
    "pedir",
    "pidiendo",
    "pagos",
    "pagar",
    "por",
    "promo",
    "promocion",
    "promociones",
    "promos",
    "que",
    "quiero",
    "quisiera",
    "rubro",
    "sacar",
    "sin",
    "sobre",
    "soy",
    "tengo",
    "tenes",
    "tenés",
    "tienen",
    "un",
    "una",
    "unos",
    "unas",
    "ver",
    "comprar",
    "comprarme",
    "compra",
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
    "contact",
    "contactless",
    "contacto",
    "consulta",
    "consultar",
    "contame",
    "cual",
    "cuales",
    "cuota",
    "cuotas",
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
    "hasta",
    "hoy",
    "interes",
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
    "pagos",
    "pagar",
    "para",
    "pedime",
    "pedir",
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
    "sacar",
    "sea",
    "sean",
    "seleccion",
    "ser",
    "si",
    "sin",
    "sincontacto",
    "solo",
    "sobre",
    "sus",
    "tengo",
    "tenes",
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
    "comprar",
    "comprarme",
    "compra",
    "necesito",
    "nueva",
    "nuevas",
    "nuevo",
    "nuevos",
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
    return [
        str(category.get("description") or "").strip()
        for category in _get_category_records()
        if str(category.get("description") or "").strip()
    ]


def resolve_benefit_category(text: str | None) -> str | None:
    record = _resolve_category_record(text)
    if not record:
        return None
    return str(record.get("description") or "").strip() or None


def infer_benefits_filters(text: str | None) -> dict[str, Any]:
    normalized = _normalize_text(text)
    category = resolve_benefit_category(normalized)
    exclude_eminent = _has_exclude_eminent_intent(normalized)
    only_eminent = _has_only_eminent_intent(normalized) and not exclude_eminent
    only_qr = any(_contains_term(normalized, pattern) for pattern in QR_PATTERNS)
    only_nfc = any(_contains_term(normalized, pattern) for pattern in NFC_PATTERNS)
    today_only = any(_contains_term(normalized, pattern) for pattern in TODAY_PATTERNS)
    every_day_only = any(_contains_term(normalized, pattern) for pattern in EVERY_DAY_PATTERNS)
    installments = _extract_installments(normalized)
    has_installments = _has_installments_intent(normalized) or installments is not None
    interest_free = _has_interest_free_intent(normalized)
    merchant_names = _detect_merchants(normalized, category=category)
    search_terms = _build_search_terms(
        normalized,
        category=category,
        merchant_names=merchant_names,
        structured_filters_present=any(
            [
                only_eminent,
                exclude_eminent,
                only_qr,
                only_nfc,
                today_only,
                every_day_only,
                has_installments,
                interest_free,
            ]
        ),
    )
    cleaned_query = " ".join(search_terms).strip()

    return {
        "category": category,
        "only_eminent": only_eminent,
        "exclude_eminent": exclude_eminent,
        "only_qr": only_qr,
        "only_nfc": only_nfc,
        "today_only": today_only,
        "every_day_only": every_day_only,
        "installments": installments,
        "has_installments": has_installments,
        "interest_free": interest_free,
        "explicit_eminent_black": _mentions_eminent_black(normalized),
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
    raw_query: str | None = None,
    only_qr: bool = False,
    only_nfc: bool = False,
    today_only: bool = False,
    every_day_only: bool = False,
    installments: int | None = None,
    has_installments: bool = False,
    interest_free: bool = False,
    search_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    effective_query = raw_query or query or ""
    inferred_filters = infer_benefits_filters(effective_query)
    canonical_category = resolve_benefit_category(category) or inferred_filters["category"]
    normalized_search_terms = [
        _normalize_text(term)
        for term in (search_terms if search_terms is not None else inferred_filters["search_terms"])
        if _normalize_text(term)
    ]
    should_exclude_eminent = exclude_eminent or inferred_filters["exclude_eminent"]
    should_only_eminent = (only_eminent or inferred_filters["only_eminent"]) and not should_exclude_eminent
    effective_only_qr = only_qr or inferred_filters["only_qr"]
    effective_only_nfc = only_nfc or inferred_filters["only_nfc"]
    effective_today_only = today_only or inferred_filters["today_only"]
    effective_every_day_only = every_day_only or inferred_filters["every_day_only"]
    requested_installments = installments if installments is not None else inferred_filters["installments"]
    requires_installments = (
        has_installments
        or inferred_filters["has_installments"]
        or requested_installments is not None
    )
    requires_interest_free = interest_free or inferred_filters["interest_free"]

    base_results = _filter_benefits(
        _iter_benefits(),
        category=canonical_category,
        search_terms=normalized_search_terms,
        exclude_eminent=should_exclude_eminent,
        only_eminent=should_only_eminent,
        only_qr=effective_only_qr,
        only_nfc=effective_only_nfc,
        today_only=effective_today_only,
        every_day_only=effective_every_day_only,
        installments=requested_installments,
        has_installments=requires_installments,
        interest_free=requires_interest_free,
        limit=limit,
    )
    if base_results:
        return base_results

    targeted_benefits = _fetch_targeted_benefits(
        raw_query=effective_query,
        category=canonical_category,
        search_terms=normalized_search_terms,
        limit=limit,
    )
    if not targeted_benefits:
        return []

    return _filter_benefits(
        targeted_benefits,
        category=canonical_category,
        search_terms=normalized_search_terms,
        exclude_eminent=should_exclude_eminent,
        only_eminent=should_only_eminent,
        only_qr=effective_only_qr,
        only_nfc=effective_only_nfc,
        today_only=effective_today_only,
        every_day_only=effective_every_day_only,
        installments=requested_installments,
        has_installments=requires_installments,
        interest_free=requires_interest_free,
        limit=limit,
    )


def _iter_benefits() -> list[dict[str, Any]]:
    cached = _read_cache(_PROMOTIONS_CACHE)
    if cached is not _CACHE_MISS:
        return list(cached)

    benefits = _fetch_live_promotions()
    _write_cache(_PROMOTIONS_CACHE, benefits, source="api")
    _merchant_index.cache_clear()
    log_step("BENEFITS_API", "Promociones base actualizadas desde la API", {"results": len(benefits)})
    return list(benefits)


def _get_category_records() -> list[dict[str, Any]]:
    cached = _read_cache(_CATEGORIES_CACHE)
    if cached is not _CACHE_MISS:
        return list(cached)

    categories = _fetch_live_categories()
    _write_cache(_CATEGORIES_CACHE, categories, source="api")
    log_step("BENEFITS_API", "Categorias actualizadas desde la API", {"results": len(categories)})
    return list(categories)


def _resolve_category_record(text: str | None) -> dict[str, Any] | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None

    categories = _get_category_records()
    for category in categories:
        description = str(category.get("description") or "").strip()
        if description and _text_matches_category(normalized, description):
            return category

    for canonical_name, aliases in CATEGORY_SYNONYMS.items():
        if not any(_text_matches_alias(normalized, alias) for alias in aliases):
            continue

        for category in categories:
            description = str(category.get("description") or "").strip()
            if _normalize_text(description) == _normalize_text(canonical_name):
                return category

    return None


def _fetch_live_categories() -> list[dict[str, Any]]:
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

    categories: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        description = str(item.get("descripcion") or "").strip()
        if not description or description in seen:
            continue

        seen.add(description)
        categories.append(
            {
                "id": _safe_int(item.get("id")),
                "description": description,
                "emoji": str(item.get("emoji") or "").strip(),
            }
        )

    return categories


def _fetch_live_promotions() -> list[dict[str, Any]]:
    return _fetch_catalog_promotions(params={})


def _fetch_catalog_promotions(params: Any) -> list[dict[str, Any]]:
    payload = _fetch_json(
        "promociones/catalogo",
        params=params,
    )
    items = ((payload.get("data") or {}).get("list")) or []
    if not isinstance(items, list):
        raise ValueError("La respuesta del catalogo de promociones no contiene una lista valida.")

    promotions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for promo in items:
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
                ]
            )

        if dedupe_key in seen_ids:
            continue

        seen_ids.add(dedupe_key)
        promotions.append(benefit)

    return promotions


def _fetch_json(path: str, *, params: Any) -> dict[str, Any]:
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
    days = str(promo.get("leyendaDiasAplicacion") or "").strip()
    additional_text = str(promo.get("adicional") or "").strip()
    model_attention_name = str(modelo_atencion.get("nombre") or "").strip()
    contact_less = bool(promo.get("contactLess"))
    eminent_flag = bool(promo.get("eminent"))
    exclusive_eminent = (
        eminent_flag
        or bool(modelo_atencion.get("exclusivo"))
        or _contains_term(model_attention_name, "eminent")
    )
    end_date = promo.get("fechaHasta")

    return {
        "id": promo.get("id"),
        "comercio": commerce,
        "beneficio": benefit_text,
        "categoria": category,
        "dias": days,
        "mediosDePago": _normalize_payment_methods(promo.get("mediosDePago") or []),
        "exclusivoEminent": exclusive_eminent,
        "eminent": eminent_flag,
        "pagoQR": bool(promo.get("pagoQR")),
        "pagoNFC": bool(promo.get("pagoNFC")) or contact_less,
        "contactLess": contact_less,
        "proximamente": bool(promo.get("proximamente")),
        "fechaHasta": end_date,
        "vigenciaHasta": end_date,
        "imagen": promo.get("imagen"),
        "tipoPromocion": promo.get("tipoPromocion"),
        "adicional": additional_text,
        "modeloAtencionNombre": model_attention_name,
        "raw_text": _build_raw_text(
            commerce,
            benefit_text,
            category,
            days,
            additional_text,
            model_attention_name,
        ),
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
            if len(token) >= 4 and token not in NORMALIZED_GENERIC_QUERY_TOKENS:
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
    structured_filters_present: bool,
) -> list[str]:
    if merchant_names:
        primary_merchant = merchant_names[0]
        return [_normalize_text(primary_merchant)]

    if category:
        return []

    if structured_filters_present:
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
        if len(alias_token) < 4 or alias_token in NORMALIZED_GENERIC_QUERY_TOKENS:
            continue

        for text_token in text_tokens:
            if len(text_token) < 4 or text_token in NORMALIZED_GENERIC_QUERY_TOKENS:
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


def _mentions_eminent_black(text: str) -> bool:
    normalized = _normalize_text(text)
    return _contains_term(normalized, "eminent black")


def _extract_installments(text: str) -> int | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None

    match = re.search(r"\b(?:hasta\s+)?(\d{1,2})\s+cuotas?\b", normalized)
    if not match:
        return None

    return _safe_int(match.group(1))


def _has_installments_intent(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    if any(_contains_term(normalized, pattern) for pattern in INSTALLMENTS_PATTERNS):
        return True

    return bool(re.search(r"\b\d{1,2}\s+cuotas?\b", normalized))


def _has_interest_free_intent(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    return any(_contains_term(normalized, pattern) for pattern in INTEREST_FREE_PATTERNS)


def _build_raw_text(*parts: Any) -> str:
    return _normalize_text(
        " ".join(str(part or "").strip() for part in parts if str(part or "").strip())
    )


def _is_eminent_benefit(benefit: dict[str, Any]) -> bool:
    return bool(
        benefit.get("exclusivoEminent")
        or benefit.get("eminent")
        or _contains_term(str(benefit.get("modeloAtencionNombre") or ""), "eminent")
    )


def _has_nfc_payment(benefit: dict[str, Any]) -> bool:
    return bool(benefit.get("pagoNFC") or benefit.get("contactLess"))


def _benefit_raw_text(benefit: dict[str, Any]) -> str:
    raw_text = str(benefit.get("raw_text") or "").strip()
    if raw_text:
        return raw_text

    return _build_raw_text(
        benefit.get("categoria"),
        benefit.get("comercio"),
        benefit.get("beneficio"),
        benefit.get("dias"),
        benefit.get("adicional"),
        " ".join(benefit.get("mediosDePago") or []),
    )


def _matches_installments(benefit: dict[str, Any], installments: int | None) -> bool:
    raw_text = _benefit_raw_text(benefit)
    if not raw_text:
        return False

    if installments is None:
        return _contains_term(raw_text, "cuotas") or _contains_term(raw_text, "cuota")

    return bool(re.search(rf"\b(?:hasta\s+)?{installments}\s+cuotas?\b", raw_text))


def _matches_interest_free(benefit: dict[str, Any]) -> bool:
    raw_text = _benefit_raw_text(benefit)
    if not raw_text:
        return False

    return _contains_term(raw_text, "sin interes")


def _matches_query_terms(benefit: dict[str, Any], search_terms: list[str]) -> bool:
    searchable_text = _benefit_raw_text(benefit)
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


def _filter_benefits(
    benefits: Iterable[dict[str, Any]],
    *,
    category: str | None,
    search_terms: list[str],
    exclude_eminent: bool,
    only_eminent: bool,
    only_qr: bool,
    only_nfc: bool,
    today_only: bool,
    every_day_only: bool,
    installments: int | None,
    has_installments: bool,
    interest_free: bool,
    limit: int,
) -> list[dict[str, Any]]:
    day_filter = _today_day_name() if today_only else None
    results: list[dict[str, Any]] = []

    for benefit in benefits:
        if category and not _benefit_matches_category(benefit.get("categoria"), category):
            continue

        if exclude_eminent and _is_eminent_benefit(benefit):
            continue

        if only_eminent and not _is_eminent_benefit(benefit):
            continue

        if only_qr and not benefit.get("pagoQR"):
            continue

        if only_nfc and not _has_nfc_payment(benefit):
            continue

        if every_day_only and not _is_every_day(benefit.get("dias")):
            continue

        if day_filter and not _matches_day(benefit.get("dias"), day_filter):
            continue

        if has_installments and not _matches_installments(benefit, installments):
            continue

        if interest_free and not _matches_interest_free(benefit):
            continue

        if search_terms and not _matches_query_terms(benefit, search_terms):
            continue

        results.append(benefit)
        if len(results) >= max(1, limit):
            break

    return results


def _fetch_targeted_benefits(
    *,
    raw_query: str,
    category: str | None,
    search_terms: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    def append_promotions(promotions: Iterable[dict[str, Any]]) -> None:
        for promotion in promotions:
            promotion_key = str(promotion.get("id") or "")
            if not promotion_key:
                promotion_key = "|".join(
                    [
                        str(promotion.get("comercio") or ""),
                        str(promotion.get("beneficio") or ""),
                        str(promotion.get("categoria") or ""),
                    ]
                )
            if promotion_key in seen_keys:
                continue
            seen_keys.add(promotion_key)
            collected.append(promotion)

    if category:
        category_promotions = _fetch_category_promotions(category)
        append_promotions(category_promotions)

    query_variants = _build_live_search_queries(
        raw_query=raw_query,
        category=category,
        search_terms=search_terms,
    )
    category_record = _resolve_category_record(category) if category else None

    for query_variant in query_variants:
        targets = _search_live_targets(query_variant, limit=8)
        for target in targets:
            if category_record and not _target_matches_category(target, category_record):
                continue

            target_promotions = _fetch_target_promotions(target)
            if not target_promotions:
                continue

            append_promotions(target_promotions)
            if len(collected) >= max(1, limit):
                return collected

    return collected


def _build_live_search_queries(
    *,
    raw_query: str,
    category: str | None,
    search_terms: list[str],
) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()

    def add_query(value: str | None) -> None:
        normalized = _normalize_text(value)
        if not normalized or normalized in seen or len(normalized) < 3:
            return
        seen.add(normalized)
        queries.append(normalized)

    if search_terms:
        add_query(" ".join(search_terms))
        for term in search_terms:
            add_query(term)

    free_text_terms = _extract_free_text_terms(raw_query)
    if free_text_terms:
        add_query(" ".join(free_text_terms))
        for term in free_text_terms:
            add_query(term)

    if category:
        add_query(category)
        for alias in CATEGORY_SYNONYMS.get(category, ()):
            add_query(alias)

    return queries


def _search_live_targets(query: str, limit: int = 8) -> list[dict[str, Any]]:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return []

    encoded_query = quote(normalized_query, safe="")
    payload = _fetch_json(
        f"buscador/search/{encoded_query}",
        params={"limit": max(1, limit)},
    )
    items = payload.get("data") or []
    if not isinstance(items, list):
        return []

    targets: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        raw_ids = item.get("ids") or []
        ids = [
            item_id
            for item_id in (_safe_int(raw_id) for raw_id in raw_ids if raw_ids)
            if item_id is not None
        ]
        target_type = str(item.get("tipo") or "").strip()
        name = str(item.get("nombre") or "").strip()
        navigation = str(item.get("navegacion") or "").strip()
        category_id = _safe_int(item.get("idCategoria"))

        if not name and not navigation:
            continue

        targets.append(
            {
                "ids": ids,
                "name": name,
                "type": target_type,
                "navigation": navigation,
                "category_id": category_id,
                "normalized_name": _normalize_text(name),
            }
        )

    return targets


def _target_matches_category(target: dict[str, Any], category_record: dict[str, Any]) -> bool:
    requested_category_id = _safe_int(category_record.get("id"))
    target_category_id = _safe_int(target.get("category_id"))

    if target_category_id is not None and requested_category_id is not None:
        return target_category_id == requested_category_id

    target_name = str(target.get("name") or "").strip()
    if target_name and _benefit_matches_category(target_name, str(category_record.get("description") or "")):
        return True

    navigation_category_id = _extract_category_id_from_navigation(str(target.get("navigation") or ""))
    if requested_category_id is not None and navigation_category_id is not None:
        return navigation_category_id == requested_category_id

    return target_category_id in {None, 0}


def _fetch_category_promotions(category: str) -> list[dict[str, Any]]:
    category_record = _resolve_category_record(category)
    if not category_record:
        return []

    category_id = _safe_int(category_record.get("id"))
    if category_id is None:
        return []

    return _fetch_catalog_promotions(
        params={
            "IdCategoria": category_id,
            "TipoPromocion": "categoria",
        }
    )


def _fetch_target_promotions(target: dict[str, Any]) -> list[dict[str, Any]]:
    target_type = str(target.get("type") or "").strip()
    if target_type.lower() in {"marca", "shopping"}:
        return _fetch_brand_promotions(target)

    category_id = _extract_category_id_from_navigation(str(target.get("navigation") or ""))
    if category_id is None:
        category_id = _safe_int(target.get("category_id"))

    if category_id is None:
        return []

    return _fetch_catalog_promotions(
        params={
            "IdCategoria": category_id,
            "TipoPromocion": "categoria",
        }
    )


def _fetch_brand_promotions(target: dict[str, Any]) -> list[dict[str, Any]]:
    brand_ids = [
        brand_id
        for brand_id in (_safe_int(raw_id) for raw_id in target.get("ids") or [])
        if brand_id is not None
    ]
    if not brand_ids:
        return []

    params: list[tuple[str, Any]] = [("IdsMarca", brand_id) for brand_id in brand_ids]
    params.append(("TipoPromocion", str(target.get("type") or "Marca").strip() or "Marca"))
    return _fetch_catalog_promotions(params=params)


def _extract_category_id_from_navigation(navigation: str) -> int | None:
    normalized_navigation = _normalize_text(navigation)
    if not normalized_navigation:
        return None

    match = re.search(r"(?:idcategoria|categoria)\s*(?:=|/)\s*(\d+)", normalized_navigation)
    if not match:
        return None

    return _safe_int(match.group(1))


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
