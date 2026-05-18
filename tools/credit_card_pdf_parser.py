from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Any


SPANISH_MONTHS = {
    "ene": 1,
    "feb": 2,
    "mar": 3,
    "abr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dic": 12,
}

AMOUNT_CAPTURE = r"-?\d{1,3}(?:\.\d{3})*(?:,\d{2})|-?\d+(?:,\d{2})"
DATE_CAPTURE = (
    r"\d{1,2}(?:[-/\s](?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic|\d{1,2}))"
    r"(?:[-/\s]\d{2,4})"
)

PURE_AMOUNT_PATTERN = re.compile(
    rf"^\$?\s*(?P<amount>{AMOUNT_CAPTURE})$",
    flags=re.IGNORECASE,
)

PURE_DATE_PATTERN = re.compile(
    rf"^(?P<date>{DATE_CAPTURE})$",
    flags=re.IGNORECASE,
)

INLINE_CURRENCY_AMOUNT_PATTERN = re.compile(
    rf"^(?P<description>.*?)(?P<currency>USD|U\$S|US\$|\$)\s*(?P<amount>{AMOUNT_CAPTURE})\s*$",
    flags=re.IGNORECASE,
)

TRAILING_CURRENCY_ONLY_PATTERN = re.compile(
    r"^(?P<description>.*?)(?P<currency>\$)\s*$",
    flags=re.IGNORECASE,
)

CARD_SUMMARY_PATTERN = re.compile(
    r"^TARJETA\s+(?P<card>\d{4})\s+Total Consumos de\s+(?P<holder>.+)$",
    flags=re.IGNORECASE,
)

TABLE_HEADERS = {
    "fecha",
    "referencia",
    "cuota",
    "comprobante",
    "pesos",
    "dolares",
}

LEGAL_SECTION_MARKERS = (
    "plan v",
    "costo financiero total",
    "tasas nominales",
    "tna",
    "tea",
    "comisiones",
    "aviso legal",
    "informacion importante",
    "legales",
    "beneficio plan",
)

TAX_KEYWORDS = (
    "impuesto",
    "interes",
    "intereses",
    "iva",
    "iibb",
    "percep",
    "percepcion",
    "db rg",
    "db.rg",
    "rg ",
    "rg5617",
    "rg 5617",
    "sellos",
    "sello",
    "ajuste",
    "cargo",
)


def parse_credit_card_statement_pdf(pdf_path: str) -> dict[str, Any]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Falta PyMuPDF. Agregá `pymupdf` a las dependencias para analizar PDFs."
        ) from exc

    page_texts: list[str] = []

    with fitz.open(pdf_path) as document:
        for page in document:
            page_texts.append(page.get_text("text"))

    return parse_credit_card_statement_text_pages(page_texts)


def parse_credit_card_statement_text_pages(
    page_texts: list[str],
) -> dict[str, Any]:
    cleaned_pages = [
        _clean_page_text(text)
        for text in page_texts
        if str(text or "").strip()
    ]

    if not cleaned_pages:
        raise ValueError("El PDF no tiene texto legible.")

    page_lines = [_extract_page_lines(text) for text in cleaned_pages]
    all_lines = [line for lines in page_lines for line in lines]
    collapsed_text = " ".join(all_lines)

    summary = _extract_summary(
        page_lines=page_lines,
        all_lines=all_lines,
        collapsed_text=collapsed_text,
    )
    document_holder = _extract_document_holder(all_lines)
    transactions, taxes_and_fees = _extract_movements(
        page_lines,
        document_holder=document_holder,
    )

    transaction_totals = _totals_by_currency(transactions)
    taxes_totals = _totals_by_currency(taxes_and_fees)

    summary["consumos"] = {
        key: value
        for key, value in transaction_totals.items()
        if value
    }
    summary["consumos_pesos"] = transaction_totals.get("ARS", 0.0)
    summary["consumos_dolares"] = transaction_totals.get("USD", 0.0)
    summary["impuestos_cargos_intereses"] = {
        key: value
        for key, value in taxes_totals.items()
        if value
    }
    summary["impuestos_cargos_intereses_pesos"] = taxes_totals.get("ARS", 0.0)
    summary["impuestos_cargos_intereses_dolares"] = taxes_totals.get("USD", 0.0)

    metadata = {
        "pages": len(cleaned_pages),
        "transactions_count": len(transactions),
        "taxes_and_fees_count": len(taxes_and_fees),
        "movements_count": len(transactions) + len(taxes_and_fees),
    }

    return {
        "summary": summary,
        "transactions": transactions,
        "taxes_and_fees": taxes_and_fees,
        "metadata": metadata,
    }


def _clean_page_text(text: str) -> str:
    cleaned = (text or "").replace("\x00", " ").replace("\xa0", " ")
    return cleaned.replace("\r", "\n")


def _extract_page_lines(page_text: str) -> list[str]:
    lines: list[str] = []

    for raw_line in page_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)

    return lines


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    lowered = without_accents.lower().replace("\xa0", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _parse_amount(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None

    cleaned = re.sub(r"[^\d,.\-]", "", raw_value or "").strip()
    if not cleaned:
        return None

    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "." in cleaned and len(cleaned.split(".")[-1]) == 3:
        cleaned = cleaned.replace(".", "")

    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def _parse_date(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None

    normalized = _normalize_text(raw_value)
    normalized = normalized.replace("/", "-").replace(" ", "-")
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")

    match = re.match(
        r"^(?P<day>\d{1,2})-(?P<month>ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic|\d{1,2})-(?P<year>\d{2,4})$",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    day = int(match.group("day"))
    month_token = match.group("month").lower()
    year = int(match.group("year"))

    if year < 100:
        year += 2000

    month = int(month_token) if month_token.isdigit() else SPANISH_MONTHS.get(month_token)
    if not month:
        return None

    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _extract_issuer(text: str) -> str:
    normalized = _normalize_text(text)
    if "visa" in normalized:
        return "VISA"
    if "mastercard" in normalized:
        return "MASTERCARD"
    return ""


def _extract_summary(
    *,
    page_lines: list[list[str]],
    all_lines: list[str],
    collapsed_text: str,
) -> dict[str, Any]:
    first_page_lines = page_lines[0] if page_lines else []
    header_dates = _extract_top_header_dates(first_page_lines)
    total_pesos, total_dolares = _extract_total_amounts(all_lines, first_page_lines)

    summary = {
        "issuer": _extract_issuer(collapsed_text),
        "total_pesos": total_pesos,
        "total_dolares": total_dolares,
        "pago_minimo": _extract_amount_after_label(
            first_page_lines,
            "pago minimo",
            max_distance=4,
        ),
        "fecha_cierre_actual": header_dates[-4] if len(header_dates) >= 4 else None,
        "fecha_vencimiento_actual": header_dates[-3] if len(header_dates) >= 3 else None,
        "proximo_cierre": header_dates[-2] if len(header_dates) >= 2 else None,
        "proximo_vencimiento": header_dates[-1] if len(header_dates) >= 1 else None,
        "limite_compra": _extract_amount_after_label(
            first_page_lines,
            "de compras en un pago y en cuotas",
            max_distance=2,
        ),
        "limite_financiacion": _extract_amount_after_label(
            first_page_lines,
            "de financiacion",
            max_distance=2,
        ),
    }

    return summary


def _extract_top_header_dates(first_page_lines: list[str]) -> list[str]:
    if not first_page_lines:
        return []

    label_index = _find_line_index(first_page_lines, "pago minimo")
    if label_index is None:
        return []

    dates = [
        _parse_date(line)
        for line in first_page_lines[:label_index]
        if _is_pure_date_line(line)
    ]
    return [value for value in dates if value]


def _extract_total_amounts(
    all_lines: list[str],
    first_page_lines: list[str],
) -> tuple[float | None, float | None]:
    for index, line in enumerate(all_lines):
        if _normalize_text(line) != "total a pagar":
            continue

        amount_lines = [
            candidate
            for candidate in all_lines[index + 1:index + 5]
            if _is_pure_amount_line(candidate)
        ]
        if amount_lines:
            ars_value = _parse_amount(amount_lines[0])
            usd_value = _parse_amount(amount_lines[1]) if len(amount_lines) > 1 else None
            return ars_value, usd_value

    if first_page_lines:
        label_index = _find_line_index(first_page_lines, "pago minimo")
        if label_index is not None:
            header_lines = first_page_lines[:label_index]
            first_date_index = next(
                (
                    index
                    for index, candidate in enumerate(header_lines)
                    if _is_pure_date_line(candidate)
                ),
                None,
            )
            if first_date_index is not None:
                amount_lines = [
                    candidate
                    for candidate in header_lines[:first_date_index]
                    if _is_pure_amount_line(candidate)
                ]
                if amount_lines:
                    ars_value = _parse_amount(amount_lines[0])
                    usd_value = _parse_amount(amount_lines[1]) if len(amount_lines) > 1 else None
                    return ars_value, usd_value

    return None, None


def _find_line_index(lines: list[str], label: str) -> int | None:
    normalized_label = _normalize_text(label)
    for index, line in enumerate(lines):
        if _normalize_text(line) == normalized_label:
            return index
    return None


def _extract_amount_after_label(
    lines: list[str],
    label: str,
    *,
    max_distance: int,
) -> float | None:
    label_index = _find_line_index(lines, label)
    if label_index is None:
        return None

    for candidate in lines[label_index + 1:label_index + 1 + max_distance]:
        if _is_pure_amount_line(candidate):
            return _parse_amount(candidate)

    return None


def _extract_document_holder(all_lines: list[str]) -> str | None:
    for index, line in enumerate(all_lines[:20]):
        normalized = _normalize_text(line)
        if "tarjeta credito" not in normalized:
            continue

        for candidate in all_lines[index + 1:index + 5]:
            if _looks_like_full_name(candidate):
                return _sanitize_holder_name(candidate)

    for line in all_lines[:20]:
        if _looks_like_full_name(line):
            return _sanitize_holder_name(line)

    return None


def _looks_like_full_name(line: str) -> bool:
    cleaned = re.sub(r"[^A-ZÁÉÍÓÚÑ ]", "", line or "").strip()
    if not cleaned or cleaned != (line or "").strip():
        return False

    words = [word for word in cleaned.split() if word]
    return len(words) >= 2


def _extract_movements(
    page_lines: list[list[str]],
    *,
    document_holder: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_lines = _collect_detail_lines(page_lines)
    transactions: list[dict[str, Any]] = []
    taxes_and_fees: list[dict[str, Any]] = []

    current_holder = document_holder
    current_card: str | None = None
    block_start_index = 0
    index = 0

    while index < len(detail_lines):
        line = detail_lines[index]
        normalized_line = _normalize_text(line)

        if normalized_line == "total a pagar":
            break

        if any(marker in normalized_line for marker in LEGAL_SECTION_MARKERS):
            break

        card_summary = _parse_card_summary_line(line)
        if card_summary:
            holder, card = card_summary
            _backfill_transactions(
                transactions,
                block_start_index,
                holder=holder,
                card=card,
            )
            block_start_index = len(transactions)
            current_holder = None
            current_card = None
            index += 1

            skipped_amounts = 0
            while index < len(detail_lines) and skipped_amounts < 2 and _is_pure_amount_line(detail_lines[index]):
                skipped_amounts += 1
                index += 1

            continue

        if not _is_pure_date_line(line):
            index += 1
            continue

        movement, next_index = _parse_movement_block(
            detail_lines,
            index,
            default_holder=current_holder,
            default_card=current_card,
        )
        if movement is None:
            index += 1
            continue

        if _is_tax_or_fee(movement["descripcion"]):
            taxes_and_fees.append(
                {
                    "fecha": movement["fecha"],
                    "descripcion": movement["descripcion"],
                    "moneda": movement["moneda"],
                    "importe": movement["importe"],
                    "tipo": _classify_tax_or_fee(movement["descripcion"]),
                }
            )
        else:
            transactions.append(movement)

        index = next_index

    _backfill_transactions(
        transactions,
        block_start_index,
        holder=current_holder,
        card=current_card,
    )

    return transactions, taxes_and_fees


def _collect_detail_lines(page_lines: list[list[str]]) -> list[str]:
    detail_lines: list[str] = []

    for lines in page_lines:
        detail_index = next(
            (
                index
                for index, line in enumerate(lines)
                if _normalize_text(line) == "detalle del consumo"
            ),
            None,
        )

        if detail_index is None:
            continue

        for line in lines[detail_index + 1:]:
            normalized = _normalize_text(line)
            if normalized in TABLE_HEADERS:
                continue

            detail_lines.append(line)

            if normalized == "total a pagar":
                return detail_lines

    return detail_lines


def _parse_card_summary_line(line: str) -> tuple[str | None, str | None] | None:
    match = CARD_SUMMARY_PATTERN.match(line.strip())
    if not match:
        return None

    return (
        _sanitize_holder_name(match.group("holder")),
        _extract_last_four(match.group("card")),
    )


def _parse_movement_block(
    lines: list[str],
    start_index: int,
    *,
    default_holder: str | None,
    default_card: str | None,
) -> tuple[dict[str, Any] | None, int]:
    parsed_date = _parse_date(lines[start_index])
    if not parsed_date:
        return None, start_index + 1

    index = start_index + 1
    description_parts: list[str] = []
    cuota: str | None = None
    comprobante: str | None = None
    amount: float | None = None
    currency: str | None = None

    while index < len(lines):
        line = lines[index].strip()
        normalized = _normalize_text(line)

        if not line:
            index += 1
            continue

        if normalized == "total a pagar" or _parse_card_summary_line(line) or _is_pure_date_line(line):
            break

        if _is_reference_marker_line(line):
            index += 1
            continue

        if cuota is None and _is_quota_line(line):
            cuota = line.strip()
            index += 1
            continue

        if comprobante is None and _is_comprobante_line(line):
            comprobante = line.strip()
            index += 1
            continue

        if amount is None and _is_pure_amount_line(line):
            amount = _parse_amount(line)
            if currency is None:
                currency = _currency_from_token("$" if "$" in line else "")
            index += 1
            continue

        description_parts.append(line)
        index += 1

    description = re.sub(r"\s{2,}", " ", " ".join(description_parts)).strip()
    if not description:
        return None, index

    description, inline_currency, inline_amount = _normalize_description(description)
    if amount is None:
        amount = inline_amount

    currency = inline_currency or currency or _currency_from_description(description) or "ARS"
    if amount is None:
        return None, index

    return (
        {
            "fecha": parsed_date,
            "descripcion": description,
            "cuota": cuota,
            "comprobante": comprobante,
            "moneda": currency,
            "importe": amount,
            "tipo": "consumo",
            "tarjeta": default_card,
            "titular": default_holder,
        },
        index,
    )


def _normalize_description(
    description: str,
) -> tuple[str, str | None, float | None]:
    cleaned = re.sub(r"\s{2,}", " ", description or "").strip()

    inline_match = INLINE_CURRENCY_AMOUNT_PATTERN.match(cleaned)
    if inline_match:
        normalized_description = inline_match.group("description").strip()
        return (
            normalized_description,
            _currency_from_token(inline_match.group("currency")),
            _parse_amount(inline_match.group("amount")),
        )

    trailing_currency_match = TRAILING_CURRENCY_ONLY_PATTERN.match(cleaned)
    if trailing_currency_match:
        return (
            trailing_currency_match.group("description").strip(),
            _currency_from_token(trailing_currency_match.group("currency")),
            None,
        )

    return cleaned, None, None


def _is_reference_marker_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    if stripped in {"*", "K", "F", "WL", "Z"}:
        return True

    return len(stripped) <= 3 and stripped.isalpha()


def _is_quota_line(line: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}/\d{1,2}", line.strip()))


def _is_comprobante_line(line: str) -> bool:
    return bool(re.fullmatch(r"\d{5,8}", line.strip()))


def _is_pure_amount_line(line: str) -> bool:
    return bool(PURE_AMOUNT_PATTERN.match(line.strip()))


def _is_pure_date_line(line: str) -> bool:
    return bool(PURE_DATE_PATTERN.match(line.strip()))


def _currency_from_token(token: str | None) -> str | None:
    normalized = _normalize_text(token or "")
    if normalized in {"u$s", "us$", "usd", "dolar", "dolares"}:
        return "USD"
    if normalized in {"ars", "peso", "pesos", "$"}:
        return "ARS"
    return None


def _currency_from_description(description: str) -> str | None:
    normalized = _normalize_text(description)
    if " usd" in f" {normalized} " or "u$s" in normalized or "us$" in normalized:
        return "USD"
    return None


def _sanitize_holder_name(raw_value: str | None) -> str | None:
    cleaned = re.sub(r"[^A-Za-zÁÉÍÓÚÑáéíóúñ ]", " ", raw_value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None

    first_name = cleaned.split()[0]
    return first_name.upper()


def _extract_last_four(raw_value: str | None) -> str | None:
    digits = re.sub(r"\D", "", raw_value or "")
    if len(digits) < 4:
        return None
    return digits[-4:]


def _backfill_transactions(
    transactions: list[dict[str, Any]],
    start_index: int,
    *,
    holder: str | None,
    card: str | None,
) -> None:
    for item in transactions[start_index:]:
        if holder and not item.get("titular"):
            item["titular"] = holder
        if card and not item.get("tarjeta"):
            item["tarjeta"] = card


def _is_tax_or_fee(description: str) -> bool:
    normalized = _normalize_text(description)
    return any(keyword in normalized for keyword in TAX_KEYWORDS)


def _classify_tax_or_fee(description: str) -> str:
    normalized = _normalize_text(description)
    if "interes" in normalized:
        return "interes"
    if "ajuste" in normalized:
        return "ajuste"
    if "cargo" in normalized:
        return "cargo"
    return "impuesto"


def _totals_by_currency(items: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {"ARS": 0.0, "USD": 0.0}

    for item in items:
        currency = str(item.get("moneda") or "ARS").upper()
        try:
            totals.setdefault(currency, 0.0)
            totals[currency] += float(item.get("importe") or 0)
        except (TypeError, ValueError):
            continue

    return {
        key: round(value, 2)
        for key, value in totals.items()
    }
