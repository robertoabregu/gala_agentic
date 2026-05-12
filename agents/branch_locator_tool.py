import json
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

import requests

from agents.state import AgentState
from observability.logger import log_step


GALICIA_BRANCHES_URL = "https://www.galicia.ar/services/sucursales"
LOCAL_FALLBACK_PATH = Path("data/branches/galicia_branches_buenos_aires.json")
MAX_BRANCHES = 5
REQUEST_TIMEOUT_SECONDS = 8

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.galicia.ar/personas/sucursales",
    "Origin": "https://www.galicia.ar",
}


def branch_locator_node(state: AgentState) -> AgentState:
    user_location = state.get("user_location") or {}

    latitude = _safe_float(user_location.get("latitude"))
    longitude = _safe_float(user_location.get("longitude"))

    if latitude is None or longitude is None:
        log_step("BRANCH_LOCATOR", "Falta ubicacion del usuario")
        return {
            **state,
            "answer": (
                "📍 Para buscar las sucursales Galicia más cercanas necesito que me compartas "
                "tu ubicación actual desde WhatsApp."
            ),
            "topic": "sucursales_cercanas",
            "needs_clarification": True,
            "missing_fields": ["user_location"],
            "pending_route": "branch_locator",
            "error": None,
        }

    branches = _fetch_live_branches(latitude, longitude)
    source_used = "endpoint_galicia"

    if not branches:
        branches = _load_local_fallback()
        source_used = "fallback_local"

    if not branches:
        log_step("BRANCH_LOCATOR", "No se obtuvieron sucursales", {"source": source_used})
        return {
            **state,
            "answer": (
                "No pude consultar las sucursales cercanas en este momento. "
                "Probá de nuevo en unos minutos."
            ),
            "topic": "sucursales_cercanas",
            "needs_clarification": False,
            "missing_fields": [],
            "pending_route": "",
            "error": None,
        }

    normalized_branches = [_normalize_branch(branch) for branch in branches]
    normalized_branches = [
        branch
        for branch in normalized_branches
        if branch.get("latitude") is not None
        and branch.get("longitude") is not None
        and branch.get("name")
        and branch.get("address")
    ]

    if not normalized_branches:
        log_step("BRANCH_LOCATOR", "Sucursales sin coordenadas validas")
        return {
            **state,
            "answer": (
                "Encontré información de sucursales, pero no pude calcular la distancia "
                "con tu ubicación actual."
            ),
            "topic": "sucursales_cercanas",
            "needs_clarification": False,
            "missing_fields": [],
            "pending_route": "",
            "error": None,
        }

    branches_with_distance = []

    for branch in normalized_branches:
        distance_km = _haversine_km(
            latitude,
            longitude,
            branch["latitude"],
            branch["longitude"],
        )
        branch["distance_km"] = distance_km
        branches_with_distance.append(branch)

    nearest_branches = sorted(
        branches_with_distance,
        key=lambda item: item["distance_km"],
    )[:MAX_BRANCHES]

    log_step(
        "BRANCH_LOCATOR",
        "Sucursales cercanas calculadas",
        {
            "source": source_used,
            "results": len(nearest_branches),
        },
    )

    return {
        **state,
        "answer": _format_answer(nearest_branches),
        "topic": "sucursales_cercanas",
        "needs_clarification": False,
        "missing_fields": [],
        "pending_route": "",
        "error": None,
    }


def _fetch_live_branches(latitude: float, longitude: float) -> list[dict[str, Any]]:
    try:
        payload = {
            "latitud": latitude,
            "longitud": longitude,
        }

        response = requests.post(
            GALICIA_BRANCHES_URL,
            json=payload,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        data = response.json()
        branches = data.get("listaPuntos", [])

        if not isinstance(branches, list):
            return []

        log_step("BRANCH_LOCATOR", "Respuesta recibida desde Galicia", {"results": len(branches)})

        return [branch for branch in branches if _is_galicia_branch(branch)]

    except Exception as exc:
        log_step("BRANCH_LOCATOR", "Error consultando endpoint Galicia", {"error": str(exc)})
        return []


def _load_local_fallback() -> list[dict[str, Any]]:
    try:
        if not LOCAL_FALLBACK_PATH.exists():
            return []

        with LOCAL_FALLBACK_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            return []

        log_step("BRANCH_LOCATOR", "Fallback local cargado", {"results": len(data)})
        return data

    except Exception as exc:
        log_step("BRANCH_LOCATOR", "Error leyendo fallback local", {"error": str(exc)})
        return []


def _is_galicia_branch(branch: dict[str, Any]) -> bool:
    description = str(branch.get("description", "")).strip().lower()
    name = str(branch.get("name", "")).strip().lower()

    if branch.get("type") not in (1, "1", None):
        return False

    return "galicia" in description or name.startswith("su ")


def _normalize_branch(branch: dict[str, Any]) -> dict[str, Any]:
    latitude = branch.get("latitude", branch.get("lat"))
    longitude = branch.get("longitude", branch.get("long"))

    return {
        "id": branch.get("id") or _make_branch_id(
            branch.get("name", ""),
            branch.get("address", ""),
        ),
        "name": str(branch.get("name", "")).strip(),
        "description": str(branch.get("description", "Banco Galicia")).strip(),
        "address": str(branch.get("address", "")).strip(),
        "city": str(branch.get("city", "")).strip(),
        "province": str(branch.get("province", "")).strip(),
        "latitude": _safe_float(latitude),
        "longitude": _safe_float(longitude),
    }


def _make_branch_id(name: str, address: str) -> str:
    value = f"{name}-{address}".lower().strip()

    replacements = {
        " ": "-",
        "/": "-",
        ".": "",
        ",": "",
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return value


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _haversine_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    earth_radius_km = 6371

    delta_latitude = radians(latitude_2 - latitude_1)
    delta_longitude = radians(longitude_2 - longitude_1)

    lat_1 = radians(latitude_1)
    lat_2 = radians(latitude_2)

    a = (
        sin(delta_latitude / 2) ** 2
        + cos(lat_1) * cos(lat_2) * sin(delta_longitude / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return earth_radius_km * c


def _format_answer(branches: list[dict[str, Any]]) -> str:
    number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]

    lines = [
        "📍 *Sucursales Galicia más cercanas*",
        "",
        "Estas son las opciones más próximas a tu ubicación:",
        "",
    ]

    for index, branch in enumerate(branches):
        number = number_emojis[index] if index < len(number_emojis) else f"{index + 1}."
        name = branch.get("name", "").strip()
        address = branch.get("address", "").strip()
        distance_text = _format_distance(branch["distance_km"])

        lines.extend(
            [
                f"{number} *{name}*",
                f"📌 {address}",
                f"🚶 Aprox. {distance_text}",
                "",
            ]
        )

    lines.append("ℹ️ Las distancias son aproximadas y pueden variar según el recorrido.")
    return "\n".join(lines).strip()


def _format_distance(distance_km: float) -> str:
    if distance_km < 1:
        meters = round(distance_km * 1000)
        return f"{meters} m"

    return f"{distance_km:.1f} km"
