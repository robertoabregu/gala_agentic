from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime
from typing import Any

import requests

from observability.logger import log_step


REQUEST_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Id_channel": "onlinebanking",
    "Id_canal": "Quiero",
    "Origin": "https://beneficios.galicia.ar",
    "Referer": "https://beneficios.galicia.ar/",
    "User-Agent": "Mozilla/5.0",
}


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


def _base_url() -> str:
    return os.getenv(
        "BENEFITS_API_BASE_URL",
        "https://loyalty.bff.bancogalicia.com.ar/api/portal/personalizacion/v1",
    ).rstrip("/")


def _request_timeout() -> float:
    return max(1.0, _get_float_env("BENEFITS_REQUEST_TIMEOUT", 8))


def _default_page_size() -> int:
    return max(1, _get_int_env("BENEFITS_LOCALES_PAGE_SIZE", 1500))


def _default_distance_from_km() -> float:
    return max(0.0, _get_float_env("BENEFITS_DISTANCE_FROM_KM", 0))


def _default_distance_to_km() -> float:
    return max(0.1, _get_float_env("BENEFITS_DISTANCE_TO_KM", 10))


def get_nearby_locales(
    latitude: float,
    longitude: float,
    *,
    page: int = 1,
    page_size: int | None = None,
    distance_from_km: float | None = None,
    distance_to_km: float | None = None,
) -> list[dict[str, Any]]:
    payload = _fetch_json(
        "locales/fisicos",
        params={
            "page": max(1, page),
            "pageSize": page_size if page_size is not None else _default_page_size(),
            "ClienteLatitud": latitude,
            "ClienteLongitud": longitude,
            "DistanciaDesde": (
                distance_from_km
                if distance_from_km is not None
                else _default_distance_from_km()
            ),
            "DistanciaHasta": (
                distance_to_km
                if distance_to_km is not None
                else _default_distance_to_km()
            ),
        },
    )

    items = _extract_list(payload)
    locales: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        normalized_local = _normalize_nearby_local(item)
        if normalized_local.get("local_id") is None:
            continue

        locales.append(normalized_local)

    log_step("BENEFITS_LOCATION_API", "Locales cercanos obtenidos", {"results": len(locales)})
    return locales


def get_local_promotions_detail(local_id: int | str) -> dict[str, Any]:
    payload = _fetch_json(f"promociones/local/{local_id}", params=None)
    detail = _extract_detail_dict(payload)
    if not isinstance(detail, dict):
        raise ValueError("La API de beneficios no devolvio un detalle valido para el local.")

    normalized_detail = _normalize_local_detail(detail, fallback_local_id=local_id)
    log_step(
        "BENEFITS_LOCATION_API",
        "Detalle de local obtenido",
        {
            "local_id": normalized_detail.get("local_id"),
            "promotions": len(normalized_detail.get("promotions") or []),
        },
    )
    return normalized_detail


def enrich_local_with_detail(
    local_summary: dict[str, Any],
    local_detail: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(local_summary)
    if not isinstance(local_detail, dict):
        merged.setdefault("promotions", [])
        merged.setdefault("address", _build_partial_address_from_summary(local_summary))
        return merged

    merged["local_id"] = local_detail.get("local_id", merged.get("local_id"))
    merged["brand"] = local_detail.get("brand") or merged.get("brand") or "Local adherido"
    merged["category"] = merged.get("category") or local_detail.get("category")
    merged["address"] = local_detail.get("address") or _build_partial_address_from_summary(local_summary)
    merged["street"] = local_detail.get("street")
    merged["number"] = local_detail.get("number")
    merged["address_note"] = local_detail.get("address_note")
    merged["city"] = local_detail.get("city") or merged.get("city")
    merged["province"] = local_detail.get("province") or merged.get("province")
    merged["postal_code"] = local_detail.get("postal_code")
    merged["country"] = local_detail.get("country")
    merged["promotions"] = list(local_detail.get("promotions") or [])
    return merged


def build_local_address(
    *,
    street: str | None,
    number: str | int | None,
    address_note: str | None,
    city: str | None,
    district: str | None = None,
    province: str | None = None,
) -> str:
    street_value = str(street or "").strip()
    number_value = str(number).strip() if number not in (None, "") else ""
    address_note_value = str(address_note or "").strip()
    city_value = str(city or "").strip()
    district_value = str(district or "").strip()
    province_value = str(province or "").strip()

    street_line = " ".join(part for part in (street_value, number_value) if part).strip()

    parts: list[str] = []
    if street_line:
        parts.append(street_line)

    if address_note_value:
        parts.append(address_note_value)

    locality = city_value or district_value or province_value
    if locality:
        parts.append(locality)

    if not parts:
        if street_value:
            parts.append(street_value)
        elif number_value:
            parts.append(number_value)
        elif district_value:
            parts.append(district_value)
        elif province_value:
            parts.append(province_value)

    return ", ".join(part for part in parts if part)


def _fetch_json(path: str, *, params: Any) -> Any:
    response = requests.get(
        f"{_base_url()}/{path.lstrip('/')}",
        headers=REQUEST_HEADERS,
        params=params,
        timeout=_request_timeout(),
    )
    response.raise_for_status()

    payload = response.json()
    if isinstance(payload, dict) and payload.get("errors"):
        raise ValueError(f"La API de beneficios devolvio errores: {payload['errors']}")
    return payload


def _extract_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    data = payload.get("data")
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("list", "items", "locales", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    for key in ("list", "items", "locales", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    return []


def _extract_detail_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            return data
        return payload

    return {}


def _normalize_nearby_local(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "local_id": _safe_int(item.get("id")),
        "brand": str(item.get("nombre") or "").strip() or "Local adherido",
        "category": str(item.get("categoriaMarca") or "").strip(),
        "city": str(item.get("localidad") or "").strip(),
        "province": str(item.get("provincia") or "").strip(),
        "distance_km": _safe_float(item.get("distancia")),
        "lat": _safe_float(item.get("latitud")),
        "lng": _safe_float(item.get("longitud")),
        "customer_lat": _safe_float(item.get("clienteLatitud")),
        "customer_lng": _safe_float(item.get("clienteLongitud")),
        "emoji": str(item.get("emoji") or "").strip(),
        "address": "",
        "promotions": [],
    }


def _normalize_local_detail(
    detail: dict[str, Any],
    *,
    fallback_local_id: int | str,
) -> dict[str, Any]:
    street = str(detail.get("calle") or "").strip()
    number = _safe_int(detail.get("numero"))
    if number is None:
        raw_number = str(detail.get("numero") or "").strip()
        number = raw_number or None

    address_note = str(detail.get("aclaracion") or "").strip() or None
    city = str(detail.get("localidadNombre") or "").strip()
    district = str(detail.get("partidoNombre") or "").strip()
    province = str(detail.get("provinciaNombre") or "").strip()

    promotions_raw = detail.get("promociones") or []
    promotions: list[dict[str, Any]] = []

    if isinstance(promotions_raw, list):
        for promotion in promotions_raw:
            if not isinstance(promotion, dict):
                continue
            promotions.append(_normalize_promotion_detail(promotion))

    return {
        "local_id": _safe_int(detail.get("idLocal")) or _safe_int(fallback_local_id),
        "brand": str(detail.get("nombreMarca") or detail.get("nombre") or "").strip(),
        "street": street or None,
        "number": number,
        "address_note": address_note,
        "postal_code": str(detail.get("codigoPostal") or "").strip() or None,
        "city": city or district or None,
        "district": district or None,
        "province": province or None,
        "country": str(detail.get("paisNombre") or "").strip() or None,
        "address": build_local_address(
            street=street,
            number=number,
            address_note=address_note,
            city=city,
            district=district,
            province=province,
        ),
        "promotions": promotions,
    }


def _normalize_promotion_detail(promotion: dict[str, Any]) -> dict[str, Any]:
    attention_model = _extract_attention_model(promotion.get("modeloAtencion"))
    payment_methods = _extract_payment_methods(promotion.get("mediosDePago") or [])
    normalized_attention_model = _normalize_text(attention_model)
    is_eminent = bool(promotion.get("eminent")) or "eminent" in normalized_attention_model

    return {
        "id": _safe_int(promotion.get("id")),
        "discount_percent": _safe_number(promotion.get("porcentajeAhorro")),
        "cap_type": str(promotion.get("tipoTope") or "").strip() or None,
        "cashback_cap": _safe_number(promotion.get("topeReintegro")),
        "additional_description": str(promotion.get("descripcionAdicional") or "").strip() or None,
        "purchase_legend": str(promotion.get("leyendaCompra") or "").strip() or None,
        "pay_legend": str(promotion.get("leyendaPaga") or "").strip() or None,
        "ready_legend": str(promotion.get("leyendaListo") or "").strip() or None,
        "days": str(promotion.get("leyendaDiasAplicacion") or "").strip() or None,
        "valid_from": _format_date(promotion.get("fechaDesde")),
        "valid_to": _format_date(promotion.get("fechaHasta")),
        "is_eminent": is_eminent,
        "attention_model": attention_model or ("Eminent" if is_eminent else None),
        "payment_methods": payment_methods,
        "payment_summary": _summarize_payment_methods(payment_methods),
        "qr": bool(promotion.get("pagoQR")),
        "nfc": bool(promotion.get("pagoNFC")),
        "contactless": bool(promotion.get("contactLess")),
        "solo_today": bool(promotion.get("soloPorHoy")),
        "on_top": bool(promotion.get("onTop")),
        "self_service": bool(promotion.get("autogestivas")),
        "coming_soon": bool(promotion.get("proximamente")),
        "installments_from": _safe_int(promotion.get("cuotaSinInteresDesde")),
        "installments_to": _safe_int(promotion.get("cuotaSinInteresHasta")),
    }


def _extract_attention_model(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("nombre", "descripcion", "detalle"):
            text = str(value.get(key) or "").strip()
            if text:
                return text

    text = str(value or "").strip()
    return text or None


def _extract_payment_methods(raw_methods: Any) -> list[str]:
    if not isinstance(raw_methods, list):
        return []

    labels: list[str] = []
    seen: set[str] = set()

    for method in raw_methods:
        label = _extract_payment_method_label(method)
        normalized_label = _normalize_text(label)
        if not normalized_label or normalized_label in seen:
            continue

        seen.add(normalized_label)
        labels.append(label)

    return labels


def _extract_payment_method_label(method: Any) -> str:
    if isinstance(method, str):
        return method.strip()

    if not isinstance(method, dict):
        return ""

    for key in ("descripcion", "nombre", "titulo", "label", "tipoMedioPago", "medioPago"):
        value = str(method.get(key) or "").strip()
        if value:
            return value

    composed_parts = [
        str(method.get("tipoTarjeta") or "").strip(),
        str(method.get("tipoCuenta") or "").strip(),
    ]
    composed = " ".join(part for part in composed_parts if part).strip()
    return composed


def _summarize_payment_methods(payment_methods: list[str]) -> str | None:
    clean_methods = [method.strip() for method in payment_methods if method.strip()]
    if not clean_methods:
        return None

    if len(clean_methods) == 1:
        return clean_methods[0]

    return f"{', '.join(clean_methods[:-1])} y {clean_methods[-1]}"


def _build_partial_address_from_summary(local_summary: dict[str, Any]) -> str:
    city = str(local_summary.get("city") or "").strip()
    province = str(local_summary.get("province") or "").strip()
    if city and province and _normalize_text(city) != _normalize_text(province):
        return f"{city}, {province}"
    return city or province


def _format_date(value: Any) -> str | None:
    if value in (None, ""):
        return None

    raw_value = str(value).strip()
    if not raw_value:
        return None

    normalized_value = raw_value.replace("Z", "+00:00")
    for parser in (
        lambda v: datetime.fromisoformat(v),
        lambda v: datetime.strptime(v, "%Y-%m-%d"),
        lambda v: datetime.strptime(v, "%d/%m/%Y"),
    ):
        try:
            parsed = parser(normalized_value)
            return parsed.strftime("%d/%m/%Y")
        except ValueError:
            continue

    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", normalized_value)
    if match:
        return f"{match.group(3)}/{match.group(2)}/{match.group(1)}"

    return raw_value


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_number(value: Any) -> int | float | None:
    number = _safe_float(value)
    if number is None:
        return None
    if number.is_integer():
        return int(number)
    return number


def _normalize_text(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    lowered = without_accents.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()
