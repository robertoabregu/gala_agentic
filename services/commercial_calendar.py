from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CALENDAR_PATH = BASE_DIR / "data" / "commercial_calendar_ar.json"
WEEKDAY_BY_NAME = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def load_commercial_calendar(path: Path | None = None) -> list[dict[str, Any]]:
    calendar_path = path or DEFAULT_CALENDAR_PATH
    try:
        payload = json.loads(calendar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(payload, list):
        return []

    return [entry for entry in payload if isinstance(entry, dict)]


def get_active_commercial_context(
    *,
    reference_date: date | None = None,
    categories: list[str] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    today = reference_date or date.today()
    normalized_categories = {_normalize_text(category) for category in categories or [] if category}
    best_match: dict[str, Any] | None = None

    for entry in load_commercial_calendar(path):
        occurrence = _resolve_relevant_occurrence(entry, today)
        if occurrence is None:
            continue

        days_until = (occurrence - today).days
        window_before_days = _safe_int(entry.get("window_before_days")) or 0
        window_after_days = _safe_int(entry.get("window_after_days")) or 0

        if days_until < -window_after_days or days_until > window_before_days:
            continue

        related_categories = [
            str(category).strip()
            for category in entry.get("related_categories") or []
            if str(category).strip()
        ]

        if normalized_categories:
            normalized_related = {_normalize_text(category) for category in related_categories}
            if normalized_related and not normalized_categories.intersection(normalized_related):
                continue

        candidate = {
            "id": str(entry.get("id") or "").strip(),
            "name": str(entry.get("name") or "").strip(),
            "days_until": days_until,
            "related_categories": related_categories,
            "response_hint": str(entry.get("response_hint") or "").strip(),
            "event_date": occurrence.isoformat(),
        }

        if best_match is None or _commercial_sort_key(candidate) < _commercial_sort_key(best_match):
            best_match = candidate

    return {"active_occasion": best_match}


def detect_explicit_occasion(
    query: str | None,
    *,
    reference_date: date | None = None,
    path: Path | None = None,
) -> dict[str, Any] | None:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return None

    today = reference_date or date.today()

    for entry in load_commercial_calendar(path):
        occasion_tokens = _build_occasion_tokens(entry)
        if not any(_contains_term(normalized_query, token) for token in occasion_tokens):
            continue

        occurrence = _resolve_upcoming_occurrence(entry, today)
        days_until = (occurrence - today).days if occurrence else None

        return {
            "id": str(entry.get("id") or "").strip(),
            "name": str(entry.get("name") or "").strip(),
            "days_until": days_until,
            "related_categories": [
                str(category).strip()
                for category in entry.get("related_categories") or []
                if str(category).strip()
            ],
            "response_hint": str(entry.get("response_hint") or "").strip(),
            "event_date": occurrence.isoformat() if occurrence else None,
        }

    return None


def _build_occasion_tokens(entry: dict[str, Any]) -> list[str]:
    tokens = {
        _normalize_text(entry.get("id")),
        _normalize_text(entry.get("name")),
    }

    normalized_name = _normalize_text(entry.get("name"))
    if normalized_name:
        tokens.add(normalized_name.replace(" de la ", " del "))
        tokens.add(normalized_name.replace(" del ", " de la "))

    return [token for token in tokens if token]


def _resolve_relevant_occurrence(entry: dict[str, Any], today: date) -> date | None:
    candidates = [
        occurrence
        for occurrence in (
            _resolve_occurrence(entry, today.year - 1),
            _resolve_occurrence(entry, today.year),
            _resolve_occurrence(entry, today.year + 1),
        )
        if occurrence is not None
    ]
    if not candidates:
        return None

    return min(candidates, key=lambda occurrence: abs((occurrence - today).days))


def _resolve_upcoming_occurrence(entry: dict[str, Any], today: date) -> date | None:
    candidates = [
        occurrence
        for occurrence in (
            _resolve_occurrence(entry, today.year),
            _resolve_occurrence(entry, today.year + 1),
        )
        if occurrence is not None and occurrence >= today - timedelta(days=7)
    ]
    if not candidates:
        return None

    return min(candidates, key=lambda occurrence: abs((occurrence - today).days))


def _resolve_occurrence(entry: dict[str, Any], year: int) -> date | None:
    rule = entry.get("rule") or {}
    if not isinstance(rule, dict):
        return None

    rule_type = str(rule.get("type") or "").strip()
    month = _safe_int(rule.get("month"))

    if rule_type == "fixed_date":
        day = _safe_int(rule.get("day"))
        if month is None or day is None:
            return None
        try:
            return date(year, month, day)
        except ValueError:
            return None

    if rule_type == "nth_weekday_of_month":
        weekday_name = str(rule.get("weekday") or "").strip().lower()
        nth = _safe_int(rule.get("nth"))
        weekday = WEEKDAY_BY_NAME.get(weekday_name)
        if month is None or nth is None or weekday is None:
            return None
        return _nth_weekday_of_month(year, month, weekday, nth)

    return None


def _nth_weekday_of_month(year: int, month: int, weekday: int, nth: int) -> date | None:
    try:
        current = date(year, month, 1)
    except ValueError:
        return None

    occurrences = 0
    while current.month == month:
        if current.weekday() == weekday:
            occurrences += 1
            if occurrences == nth:
                return current
        current += timedelta(days=1)

    return None


def _commercial_sort_key(occasion: dict[str, Any]) -> tuple[int, int, str]:
    days_until = _safe_int(occasion.get("days_until")) or 0
    return (abs(days_until), 0 if days_until >= 0 else 1, str(occasion.get("id") or ""))


def _contains_term(text: str, term: str) -> bool:
    return f" {term} " in f" {text} "


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
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
