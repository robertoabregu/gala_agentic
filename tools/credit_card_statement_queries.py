from __future__ import annotations

import re
import unicodedata
from typing import Any


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    lowered = without_accents.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _iter_transactions(statement: dict[str, Any]) -> list[dict[str, Any]]:
    items = statement.get("transactions") or []
    return [item for item in items if isinstance(item, dict)]


def _iter_taxes_and_fees(statement: dict[str, Any]) -> list[dict[str, Any]]:
    items = statement.get("taxes_and_fees") or []
    return [item for item in items if isinstance(item, dict)]


def _normalize_currency(currency: str | None) -> str | None:
    if not currency:
        return None

    normalized = _normalize_text(currency)
    if normalized in {"usd", "u s d", "u s", "dolar", "dolares", "dolar estadounidense"}:
        return "USD"
    if normalized in {"ars", "peso", "pesos", "$"}:
        return "ARS"
    return normalized.upper()


def _matches_currency(item: dict[str, Any], currency: str | None) -> bool:
    expected = _normalize_currency(currency)
    if not expected:
        return True
    return _normalize_currency(str(item.get("moneda") or "")) == expected


def _matches_titular(item: dict[str, Any], titular: str | None) -> bool:
    if not titular:
        return True
    return _normalize_text(str(item.get("titular") or "")) == _normalize_text(titular)


def _matches_merchant(item: dict[str, Any], merchant: str | None) -> bool:
    if not merchant:
        return True
    return _normalize_text(merchant) in _normalize_text(str(item.get("descripcion") or ""))


def _sum_amounts(items: list[dict[str, Any]]) -> float:
    total = 0.0
    for item in items:
        try:
            total += float(item.get("importe") or 0)
        except (TypeError, ValueError):
            continue
    return round(total, 2)


def get_largest_transaction(
    statement: dict[str, Any],
    currency: str | None = None,
) -> dict[str, Any]:
    matches = list_transactions(statement, currency=currency)["transactions"]
    if not matches:
        return {"transaction": None, "count": 0}

    if currency is None:
        currencies = {
            _normalize_currency(str(item.get("moneda") or ""))
            for item in matches
        }
        currencies.discard(None)

        if len(currencies) > 1:
            by_currency: dict[str, dict[str, Any]] = {}
            for current_currency in sorted(currencies):
                current_matches = [
                    item
                    for item in matches
                    if _normalize_currency(str(item.get("moneda") or "")) == current_currency
                ]
                if not current_matches:
                    continue
                by_currency[current_currency] = max(
                    current_matches,
                    key=lambda item: float(item.get("importe") or 0),
                )

            return {
                "transaction": None,
                "count": len(matches),
                "by_currency": by_currency,
            }

    transaction = max(
        matches,
        key=lambda item: float(item.get("importe") or 0),
    )
    return {
        "transaction": transaction,
        "count": len(matches),
    }


def list_transactions(
    statement: dict[str, Any],
    currency: str | None = None,
    titular: str | None = None,
    merchant: str | None = None,
) -> dict[str, Any]:
    matches = [
        item
        for item in _iter_transactions(statement)
        if _matches_currency(item, currency)
        and _matches_titular(item, titular)
        and _matches_merchant(item, merchant)
    ]

    return {
        "transactions": matches,
        "count": len(matches),
        "total": _sum_amounts(matches),
        "currency": _normalize_currency(currency),
        "titular": titular,
        "merchant": merchant,
    }


def get_total_by_currency(statement: dict[str, Any], currency: str) -> dict[str, Any]:
    normalized_currency = _normalize_currency(currency)
    listed = list_transactions(statement, currency=normalized_currency)
    return {
        "currency": normalized_currency,
        "total": listed["total"],
        "count": listed["count"],
    }


def get_total_taxes_and_fees(statement: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, float] = {"ARS": 0.0, "USD": 0.0}
    items = _iter_taxes_and_fees(statement)

    for item in items:
        currency = _normalize_currency(str(item.get("moneda") or "")) or "ARS"
        try:
            grouped.setdefault(currency, 0.0)
            grouped[currency] += float(item.get("importe") or 0)
        except (TypeError, ValueError):
            continue

    return {
        "totals": {
            currency: round(amount, 2)
            for currency, amount in grouped.items()
            if amount
        },
        "count": len(items),
        "items": items,
    }


def list_taxes_and_fees(statement: dict[str, Any]) -> dict[str, Any]:
    items = _iter_taxes_and_fees(statement)
    return {
        "items": items,
        "count": len(items),
        "totals": get_total_taxes_and_fees(statement)["totals"],
    }


def list_installments(statement: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in _iter_transactions(statement)
        if str(item.get("cuota") or "").strip()
    ]
    return {
        "transactions": matches,
        "count": len(matches),
        "total_ars": _sum_amounts(
            [item for item in matches if _normalize_currency(item.get("moneda")) == "ARS"]
        ),
        "total_usd": _sum_amounts(
            [item for item in matches if _normalize_currency(item.get("moneda")) == "USD"]
        ),
    }


def count_transactions(
    statement: dict[str, Any],
    currency: str | None = None,
) -> dict[str, Any]:
    listed = list_transactions(statement, currency=currency)
    return {
        "count": listed["count"],
        "currency": listed["currency"],
    }


def search_transactions(statement: dict[str, Any], text: str) -> dict[str, Any]:
    normalized_search = _normalize_text(text)
    matches = [
        item
        for item in _iter_transactions(statement)
        if normalized_search
        and normalized_search in _normalize_text(str(item.get("descripcion") or ""))
    ]

    totals: dict[str, float] = {}
    for item in matches:
        currency = _normalize_currency(str(item.get("moneda") or "")) or "ARS"
        totals.setdefault(currency, 0.0)
        try:
            totals[currency] += float(item.get("importe") or 0)
        except (TypeError, ValueError):
            continue

    return {
        "transactions": matches,
        "count": len(matches),
        "search_text": text,
        "totals": {
            currency: round(amount, 2)
            for currency, amount in totals.items()
        },
    }
