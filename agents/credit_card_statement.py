from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from typing import Any

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - fallback solo para tests sin dependencias
    ChatOpenAI = Any  # type: ignore[misc,assignment]

from agents.state import AgentState
from observability.logger import log_step
from services.twilio_media import (
    TwilioMediaError,
    cleanup_temp_file,
    download_twilio_pdf_to_tempfile,
    looks_like_pdf_media,
)
from tools.credit_card_pdf_parser import parse_credit_card_statement_pdf
from tools.credit_card_statement_queries import (
    count_transactions,
    get_largest_transaction,
    get_total_by_currency,
    get_total_taxes_and_fees,
    list_installments,
    list_taxes_and_fees,
    list_transactions,
    search_transactions,
)


TOPIC = "resumen_tarjeta"
TOOL_NAME = "credit_card_statement"
MAX_LIST_ITEMS = 12

SUPPORTED_ACTIONS = {
    "initial_summary",
    "largest_transaction",
    "list_transactions",
    "total_by_currency",
    "total_taxes_and_fees",
    "list_taxes_and_fees",
    "list_installments",
    "count_transactions",
    "search_transactions",
    "unknown",
}

ACTION_INTERPRETER_SYSTEM_PROMPT = """
Sos un interprete de intencion para consultas sobre un resumen de tarjeta de credito ya parseado.
No calcules importes.
No inventes movimientos.
No respondas en lenguaje natural.
Devolve solo JSON valido.

Formato:
{
  "action": "list_transactions",
  "filters": {
    "currency": "USD",
    "merchant": null,
    "titular": null,
    "text": null
  }
}

Acciones permitidas:
- initial_summary
- largest_transaction
- list_transactions
- total_by_currency
- total_taxes_and_fees
- list_taxes_and_fees
- list_installments
- count_transactions
- search_transactions
- unknown

Usa search_transactions cuando el usuario nombre un comercio o texto puntual del movimiento.
Usa total_by_currency solo si pregunta por la suma o total de consumos en pesos o dolares.
Usa total_taxes_and_fees solo si pregunta por impuestos, cargos o intereses.
""".strip()

SEARCH_STOPWORDS = {
    "cuanto",
    "cuanta",
    "cuantos",
    "cuantas",
    "gasto",
    "gastos",
    "consumo",
    "consumos",
    "resumen",
    "tarjeta",
    "analizado",
    "previamente",
    "mostrame",
    "mostrarme",
    "mostra",
    "lista",
    "lista",
    "quiero",
    "tengo",
    "hice",
    "hizo",
    "usd",
    "dolar",
    "dolares",
    "pesos",
    "ars",
    "impuestos",
    "impuesto",
    "cargos",
    "cargo",
    "intereses",
    "interes",
    "cuotas",
    "cuota",
    "todos",
    "todas",
    "del",
    "de",
    "en",
    "la",
    "el",
    "los",
    "las",
    "mi",
    "me",
    "por",
    "fue",
    "mas",
    "grande",
}


def credit_card_statement_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
) -> AgentState:
    question = (state.get("standalone_question") or state.get("question") or "").strip()
    media = state.get("media") or {}
    statement = _load_statement(state)

    if media:
        if not looks_like_pdf_media(media):
            log_step(
                "CREDIT_CARD_STATEMENT",
                "Adjunto descartado por content type no PDF",
                {"content_type": str(media.get("content_type") or "")},
            )
            return _build_clarification_state(
                state,
                answer=(
                    "Necesito que me adjuntes el *PDF* del resumen de tarjeta para poder analizarlo. 📄"
                ),
            )

        temp_path = None

        try:
            temp_path = download_twilio_pdf_to_tempfile(media)
            parsed_statement = parse_credit_card_statement_pdf(str(temp_path))

            log_step(
                "CREDIT_CARD_STATEMENT",
                "Resumen parseado correctamente",
                _safe_statement_metadata(parsed_statement, action="initial_summary"),
            )

            return {
                **state,
                "route": "credit_card_statement",
                "topic": TOPIC,
                "tool_name": TOOL_NAME,
                "tool_input": {
                    "has_media": True,
                    "content_type": str(media.get("content_type") or ""),
                },
                "tool_output": _safe_statement_metadata(
                    parsed_statement,
                    action="initial_summary",
                ),
                "credit_card_statement": parsed_statement,
                "needs_clarification": False,
                "missing_fields": [],
                "answer": _format_initial_summary(parsed_statement),
                "error": None,
            }
        except TwilioMediaError as exc:
            log_step(
                "CREDIT_CARD_STATEMENT",
                "Error descargando PDF de Twilio",
                {"error": str(exc)},
            )
            return _build_clarification_state(
                state,
                answer=str(exc),
            )
        except Exception as exc:
            log_step(
                "CREDIT_CARD_STATEMENT",
                "Error parseando resumen",
                {"error": str(exc)},
            )
            return _build_clarification_state(
                state,
                answer=(
                    "No pude leer ese archivo como un resumen de tarjeta Galicia/Visa. "
                    "Si querés, reenviame el *PDF* completo y legible. 📄"
                ),
            )
        finally:
            cleanup_temp_file(temp_path)

    if not statement:
        return _build_clarification_state(
            state,
            answer=(
                "Para analizar tu resumen de tarjeta necesito que me adjuntes el *PDF* por WhatsApp. 📄"
            ),
        )

    action_payload = _resolve_action(question, statement, llm=llm)
    answer = _run_statement_action(
        statement=statement,
        action_payload=action_payload,
        question=question,
    )

    log_step(
        "CREDIT_CARD_STATEMENT",
        "Consulta resuelta sobre resumen guardado",
        {
            **_safe_statement_metadata(statement, action=action_payload["action"]),
            "currency": action_payload["filters"].get("currency"),
            "titular": action_payload["filters"].get("titular"),
            "merchant": action_payload["filters"].get("merchant"),
        },
    )

    return {
        **state,
        "route": "credit_card_statement",
        "topic": TOPIC,
        "tool_name": TOOL_NAME,
        "tool_input": {
            "has_media": False,
            "has_statement": True,
        },
        "tool_output": {
            "action": action_payload["action"],
            "filters": {
                "currency": action_payload["filters"].get("currency"),
                "titular": action_payload["filters"].get("titular"),
                "merchant": action_payload["filters"].get("merchant"),
            },
            **_safe_statement_metadata(statement),
        },
        "credit_card_statement": statement,
        "needs_clarification": False,
        "missing_fields": [],
        "answer": answer,
        "error": None,
    }


def _build_clarification_state(state: AgentState, *, answer: str) -> AgentState:
    return {
        **state,
        "route": "credit_card_statement",
        "topic": TOPIC,
        "tool_name": TOOL_NAME,
        "tool_input": {
            "has_media": bool(state.get("media")),
        },
        "tool_output": {
            "action": "missing_pdf",
        },
        "needs_clarification": True,
        "missing_fields": ["pdf"],
        "answer": answer,
        "error": None,
    }


def _load_statement(state: AgentState) -> dict[str, Any]:
    current_statement = state.get("credit_card_statement")
    if isinstance(current_statement, dict) and current_statement:
        return current_statement

    memory = state.get("memory") or {}
    remembered_statement = memory.get("credit_card_statement")
    if isinstance(remembered_statement, dict):
        return remembered_statement

    return {}


def _safe_statement_metadata(
    statement: dict[str, Any],
    *,
    action: str | None = None,
) -> dict[str, Any]:
    metadata = statement.get("metadata") or {}
    return {
        "action": action or "",
        "pages": int(metadata.get("pages") or 0),
        "transactions_count": int(metadata.get("transactions_count") or 0),
        "taxes_and_fees_count": int(metadata.get("taxes_and_fees_count") or 0),
        "movements_count": int(metadata.get("movements_count") or 0),
    }


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


def _format_amount(amount: float | None, currency: str) -> str:
    if amount is None:
        return "No lo pude extraer"

    formatted = f"{amount:,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")

    if currency == "USD":
        return f"USD {formatted}"

    return f"$ {formatted}"


def _format_date(iso_value: str | None) -> str:
    if not iso_value:
        return "No lo pude extraer"

    try:
        parsed = date.fromisoformat(iso_value)
    except ValueError:
        return iso_value

    month_labels = {
        1: "Ene",
        2: "Feb",
        3: "Mar",
        4: "Abr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Ago",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dic",
    }
    return f"{parsed.day:02d}-{month_labels[parsed.month]}-{parsed.year % 100:02d}"


def _format_initial_summary(statement: dict[str, Any]) -> str:
    summary = statement.get("summary") or {}
    metadata = statement.get("metadata") or {}

    lines = [
        "📄 Analicé tu resumen de tarjeta.",
        "",
        "*Total a pagar:*",
        f"• Pesos: {_format_amount(summary.get('total_pesos'), 'ARS')}",
        f"• Dólares: {_format_amount(summary.get('total_dolares'), 'USD')}",
    ]

    pago_minimo = summary.get("pago_minimo")
    if pago_minimo is not None:
        lines.append(f"• Pago mínimo: {_format_amount(pago_minimo, 'ARS')}")

    lines.extend(
        [
            "",
            "*Fechas importantes:*",
            f"• Vencimiento: {_format_date(summary.get('fecha_vencimiento_actual'))}",
            f"• Próximo cierre: {_format_date(summary.get('proximo_cierre'))}",
            f"• Próximo vencimiento: {_format_date(summary.get('proximo_vencimiento'))}",
            "",
            "*Límites:*",
            f"• Compra en un pago/cuotas: {_format_amount(summary.get('limite_compra'), 'ARS')}",
            f"• Financiación: {_format_amount(summary.get('limite_financiacion'), 'ARS')}",
            "",
            "*Detalle:*",
            (
                f"Detecté {int(metadata.get('transactions_count') or 0)} consumos y "
                f"{int(metadata.get('taxes_and_fees_count') or 0)} cargos/impuestos/ajustes."
            ),
            "",
            "No incluí avisos legales ni textos informativos del final del resumen.",
        ]
    )

    return "\n".join(lines)


def _resolve_action(
    question: str,
    statement: dict[str, Any],
    *,
    llm: ChatOpenAI | None,
) -> dict[str, Any]:
    normalized_question = _normalize_text(question)
    currency = _extract_currency(normalized_question)
    titular = _extract_holder(statement, normalized_question)
    merchant = _extract_search_text(statement, normalized_question)

    if _is_initial_summary_request(normalized_question):
        return _action_payload("initial_summary")

    if _is_largest_transaction_request(normalized_question):
        return _action_payload(
            "largest_transaction",
            currency=currency,
            titular=titular,
            merchant=merchant,
        )

    if _is_total_taxes_request(normalized_question):
        return _action_payload("total_taxes_and_fees")

    if _is_list_taxes_request(normalized_question):
        return _action_payload("list_taxes_and_fees")

    if "cuota" in normalized_question:
        return _action_payload("list_installments")

    if _is_count_request(normalized_question):
        return _action_payload("count_transactions", currency=currency)

    if currency and _is_total_like_request(normalized_question):
        return _action_payload("total_by_currency", currency=currency)

    if merchant and (_is_total_like_request(normalized_question) or "gaste" in normalized_question):
        return _action_payload(
            "search_transactions",
            currency=currency,
            titular=titular,
            merchant=merchant,
            text=merchant,
        )

    if titular:
        return _action_payload(
            "list_transactions",
            currency=currency,
            titular=titular,
        )

    if currency:
        return _action_payload("list_transactions", currency=currency)

    if merchant:
        return _action_payload(
            "search_transactions",
            currency=currency,
            titular=titular,
            merchant=merchant,
            text=merchant,
        )

    llm_payload = _resolve_action_with_llm(
        question=question,
        statement=statement,
        llm=llm,
    )
    if llm_payload:
        return llm_payload

    return _action_payload("unknown")


def _action_payload(
    action: str,
    *,
    currency: str | None = None,
    titular: str | None = None,
    merchant: str | None = None,
    text: str | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "filters": {
            "currency": currency,
            "titular": titular,
            "merchant": merchant,
            "text": text,
        },
    }


def _extract_currency(normalized_question: str) -> str | None:
    if "usd" in normalized_question or "dolar" in normalized_question or "dolares" in normalized_question:
        return "USD"
    if "peso" in normalized_question or "pesos" in normalized_question or "ars" in normalized_question:
        return "ARS"
    return None


def _extract_holder(statement: dict[str, Any], normalized_question: str) -> str | None:
    holders = {
        str(item.get("titular") or "").strip()
        for item in statement.get("transactions", [])
        if str(item.get("titular") or "").strip()
    }

    for holder in holders:
        if _normalize_text(holder) and _normalize_text(holder) in normalized_question:
            return holder

    return None


def _extract_search_text(statement: dict[str, Any], normalized_question: str) -> str | None:
    quoted_match = re.search(r"['\"]([^'\"]{3,40})['\"]", normalized_question)
    if quoted_match:
        return quoted_match.group(1).strip()

    candidate_descriptions = [
        _normalize_text(str(item.get("descripcion") or ""))
        for item in statement.get("transactions", [])
        if str(item.get("descripcion") or "").strip()
    ]

    tokens = [
        token
        for token in normalized_question.split()
        if len(token) >= 3 and token not in SEARCH_STOPWORDS
    ]

    if not tokens:
        return None

    phrases: list[str] = []
    for size in (3, 2, 1):
        for index in range(0, len(tokens) - size + 1):
            phrases.append(" ".join(tokens[index:index + size]))

    for phrase in phrases:
        if any(phrase in description for description in candidate_descriptions):
            return phrase

    return None


def _is_initial_summary_request(normalized_question: str) -> bool:
    if any(
        pattern in normalized_question
        for pattern in (
            "cuanto",
            "cuanta",
            "cual",
            "cuales",
            "mostrame",
            "mostrar",
            "lista",
            "listar",
            "gaste",
            "suma",
            "mas grande",
            "mayor",
            "impuesto",
            "cargo",
            "interes",
            "cuota",
            "cuantos",
            "cuantas",
            "consumo",
            "consumos",
            "gasto",
            "gastos",
            "dolar",
            "dolares",
            "pesos",
            "usd",
        )
    ):
        return False

    return any(
        pattern in normalized_question
        for pattern in (
            "analiza mi resumen",
            "analizar mi resumen",
            "quiero analizar mi resumen",
            "te paso mi resumen",
            "resumen de visa",
            "analizar resumen de tarjeta adjunto",
            "resumen de tarjeta",
            "resumen de la tarjeta",
        )
    )


def _is_largest_transaction_request(normalized_question: str) -> bool:
    return any(
        pattern in normalized_question
        for pattern in (
            "mas grande",
            "mayor gasto",
            "gasto mas grande",
            "consumo mas grande",
            "el mas grande",
            "la compra mas grande",
        )
    )


def _is_total_like_request(normalized_question: str) -> bool:
    return any(
        pattern in normalized_question
        for pattern in (
            "cuanto",
            "cuanta",
            "suma",
            "suman",
            "suman los",
            "total",
            "gaste",
            "gastado",
        )
    )


def _is_total_taxes_request(normalized_question: str) -> bool:
    if not any(term in normalized_question for term in ("impuesto", "impuestos", "cargo", "cargos", "interes", "intereses")):
        return False

    return _is_total_like_request(normalized_question) or "cobraron" in normalized_question


def _is_list_taxes_request(normalized_question: str) -> bool:
    return any(term in normalized_question for term in ("impuesto", "impuestos", "cargo", "cargos", "interes", "intereses")) and any(
        verb in normalized_question
        for verb in ("mostra", "mostrame", "listar", "lista", "cuales", "cuáles")
    )


def _is_count_request(normalized_question: str) -> bool:
    return any(
        pattern in normalized_question
        for pattern in (
            "cuantos gastos",
            "cuantos consumos",
            "cuantos movimientos",
            "cantidad de gastos",
            "cantidad de consumos",
            "cuantos hubo",
        )
    )


def _resolve_action_with_llm(
    *,
    question: str,
    statement: dict[str, Any],
    llm: ChatOpenAI | None,
) -> dict[str, Any] | None:
    if llm is None:
        return None

    holders = sorted(
        {
            str(item.get("titular") or "").strip()
            for item in statement.get("transactions", [])
            if str(item.get("titular") or "").strip()
        }
    )

    try:
        response = llm.invoke(
            [
                ("system", ACTION_INTERPRETER_SYSTEM_PROMPT),
                (
                    "user",
                    json.dumps(
                        {
                            "question": question,
                            "holders": holders,
                            "summary": {
                                "transactions_count": int(
                                    (statement.get("metadata") or {}).get("transactions_count") or 0
                                ),
                                "taxes_and_fees_count": int(
                                    (statement.get("metadata") or {}).get("taxes_and_fees_count") or 0
                                ),
                            },
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
        parsed = _parse_action_response(getattr(response, "content", ""))
    except Exception:
        return None

    if not parsed:
        return None

    return parsed


def _parse_action_response(content: Any) -> dict[str, Any] | None:
    if isinstance(content, list):
        text = "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
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

    if not isinstance(parsed, dict):
        return None

    action = str(parsed.get("action") or "").strip()
    if action not in SUPPORTED_ACTIONS:
        return None

    filters = parsed.get("filters") if isinstance(parsed.get("filters"), dict) else {}
    return _action_payload(
        action,
        currency=_safe_string(filters.get("currency")),
        titular=_safe_string(filters.get("titular")),
        merchant=_safe_string(filters.get("merchant")),
        text=_safe_string(filters.get("text")),
    )


def _safe_string(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _run_statement_action(
    *,
    statement: dict[str, Any],
    action_payload: dict[str, Any],
    question: str,
) -> str:
    action = action_payload["action"]
    filters = action_payload["filters"]
    normalized_question = _normalize_text(question)

    if action == "initial_summary":
        return _format_initial_summary(statement)

    if action == "largest_transaction":
        result = get_largest_transaction(statement, currency=filters.get("currency"))
        return _format_largest_transaction_answer(result, currency=filters.get("currency"))

    if action == "list_transactions":
        result = list_transactions(
            statement,
            currency=filters.get("currency"),
            titular=filters.get("titular"),
            merchant=filters.get("merchant"),
        )
        return _format_transactions_list_answer(result)

    if action == "total_by_currency":
        currency = filters.get("currency") or "ARS"
        result = get_total_by_currency(statement, currency)
        return _format_total_by_currency_answer(result)

    if action == "total_taxes_and_fees":
        result = get_total_taxes_and_fees(statement)
        return _format_total_taxes_answer(result)

    if action == "list_taxes_and_fees":
        result = list_taxes_and_fees(statement)
        return _format_taxes_list_answer(result)

    if action == "list_installments":
        result = list_installments(statement)
        return _format_installments_answer(result)

    if action == "count_transactions":
        result = count_transactions(statement, currency=filters.get("currency"))
        return _format_count_answer(result)

    if action == "search_transactions":
        search_text = filters.get("text") or filters.get("merchant") or ""
        result = search_transactions(statement, search_text)
        return _format_search_answer(result, normalized_question)

    return (
        "Puedo ayudarte con consumos, impuestos, gastos en pesos o dólares, cuotas y el gasto más grande del resumen que analizamos."
    )


def _format_largest_transaction_answer(
    result: dict[str, Any],
    *,
    currency: str | None,
) -> str:
    by_currency = result.get("by_currency")
    if isinstance(by_currency, dict) and by_currency:
        lines = [
            "No comparo pesos y dólares directamente, pero estos son los más grandes que encontré:",
            "",
        ]

        ars_transaction = by_currency.get("ARS")
        if isinstance(ars_transaction, dict):
            lines.append(
                f"• En pesos: {ars_transaction.get('descripcion')} — {_format_amount(ars_transaction.get('importe'), 'ARS')}"
            )

        usd_transaction = by_currency.get("USD")
        if isinstance(usd_transaction, dict):
            lines.append(
                f"• En dólares: {usd_transaction.get('descripcion')} — {_format_amount(usd_transaction.get('importe'), 'USD')}"
            )

        return "\n".join(lines)

    transaction = result.get("transaction")
    if not isinstance(transaction, dict):
        if currency == "USD":
            return "No encontré consumos en dólares en el resumen analizado."
        if currency == "ARS":
            return "No encontré consumos en pesos en el resumen analizado."
        return "No encontré consumos para calcular el gasto más grande."

    lines = [
        "El consumo más grande que encontré fue:",
        "",
        f"• {transaction.get('descripcion')} — {_format_amount(transaction.get('importe'), transaction.get('moneda', 'ARS'))}",
        f"• Fecha: {_format_date(transaction.get('fecha'))}",
    ]

    if transaction.get("titular"):
        lines.append(f"• Titular: {transaction['titular']}")

    if transaction.get("cuota"):
        lines.append(f"• Cuota: {transaction['cuota']}")

    return "\n".join(lines)


def _format_transactions_list_answer(result: dict[str, Any]) -> str:
    transactions = result.get("transactions") or []
    if not transactions:
        filter_label = _describe_transaction_filters(result)
        return f"No encontré consumos {filter_label} en el resumen analizado."

    header = _build_transaction_list_header(result)
    lines = [header, ""]

    for item in transactions[:MAX_LIST_ITEMS]:
        lines.append(
            f"• {item.get('descripcion')} — {_format_amount(item.get('importe'), item.get('moneda', 'ARS'))}"
        )

    if len(transactions) > MAX_LIST_ITEMS:
        lines.extend(
            [
                "",
                f"Mostré {MAX_LIST_ITEMS} de {len(transactions)} consumos detectados.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                f"Detecté {len(transactions)} consumos{_count_suffix(result)}.",
            ]
        )

    return "\n".join(lines)


def _format_total_by_currency_answer(result: dict[str, Any]) -> str:
    currency = result.get("currency") or "ARS"
    label = "dólares" if currency == "USD" else "pesos"
    return (
        f"Los consumos en {label} suman *{_format_amount(result.get('total'), currency)}*.\n\n"
        f"Detecté {int(result.get('count') or 0)} consumos en {currency}."
    )


def _format_total_taxes_answer(result: dict[str, Any]) -> str:
    totals = result.get("totals") or {}
    if not totals:
        return "No encontré impuestos, cargos ni intereses en el resumen analizado."

    lines = ["En impuestos/cargos detecté aproximadamente:"]
    if "ARS" in totals:
        lines.append(f"• *{_format_amount(totals['ARS'], 'ARS')}*")
    if "USD" in totals:
        lines.append(f"• *{_format_amount(totals['USD'], 'USD')}*")

    descriptions = []
    for item in result.get("items", [])[:5]:
        description = str(item.get("descripcion") or "").strip()
        if description and description not in descriptions:
            descriptions.append(description)

    if descriptions:
        lines.extend(["", "Incluye:"])
        for description in descriptions:
            lines.append(f"• {description}")

    return "\n".join(lines)


def _format_taxes_list_answer(result: dict[str, Any]) -> str:
    items = result.get("items") or []
    if not items:
        return "No encontré impuestos, cargos ni intereses en el resumen analizado."

    lines = ["Estos son los impuestos/cargos que encontré:", ""]
    for item in items[:MAX_LIST_ITEMS]:
        lines.append(
            f"• {item.get('descripcion')} — {_format_amount(item.get('importe'), item.get('moneda', 'ARS'))}"
        )

    return "\n".join(lines)


def _format_installments_answer(result: dict[str, Any]) -> str:
    transactions = result.get("transactions") or []
    if not transactions:
        return "No encontré consumos en cuotas en el resumen analizado."

    lines = ["Estos son los consumos en cuotas que encontré:", ""]
    for item in transactions[:MAX_LIST_ITEMS]:
        cuota = str(item.get("cuota") or "").strip()
        lines.append(
            f"• {item.get('descripcion')} — cuota {cuota} — {_format_amount(item.get('importe'), item.get('moneda', 'ARS'))}"
        )

    lines.extend(
        [
            "",
            f"Detecté {len(transactions)} consumos en cuotas.",
        ]
    )
    return "\n".join(lines)


def _format_count_answer(result: dict[str, Any]) -> str:
    currency = result.get("currency")
    if currency == "USD":
        return f"Detecté *{int(result.get('count') or 0)} consumos en USD*."
    if currency == "ARS":
        return f"Detecté *{int(result.get('count') or 0)} consumos en pesos*."
    return f"Detecté *{int(result.get('count') or 0)} consumos*."


def _format_search_answer(result: dict[str, Any], normalized_question: str) -> str:
    transactions = result.get("transactions") or []
    search_text = str(result.get("search_text") or "").strip()

    if not transactions:
        return f"No encontré consumos que coincidan con *{search_text}* en el resumen analizado."

    if _is_total_like_request(normalized_question):
        totals = result.get("totals") or {}
        lines = [f"Encontré {len(transactions)} consumos relacionados con *{search_text}*."]
        if "ARS" in totals:
            lines.append(f"• Total en pesos: *{_format_amount(totals['ARS'], 'ARS')}*")
        if "USD" in totals:
            lines.append(f"• Total en dólares: *{_format_amount(totals['USD'], 'USD')}*")
        return "\n".join(lines)

    lines = [f"Estos son los consumos que encontré para *{search_text}*:", ""]
    for item in transactions[:MAX_LIST_ITEMS]:
        lines.append(
            f"• {item.get('descripcion')} — {_format_amount(item.get('importe'), item.get('moneda', 'ARS'))}"
        )

    return "\n".join(lines)


def _build_transaction_list_header(result: dict[str, Any]) -> str:
    currency = result.get("currency")
    titular = result.get("titular")
    merchant = result.get("merchant")

    if currency == "USD":
        return "Estos son los consumos en dólares que encontré:"
    if currency == "ARS":
        return "Estos son los consumos en pesos que encontré:"
    if titular:
        return f"Estos son los consumos de {titular} que encontré:"
    if merchant:
        return f"Estos son los consumos vinculados a {merchant} que encontré:"
    return "Estos son los consumos que encontré:"


def _describe_transaction_filters(result: dict[str, Any]) -> str:
    currency = result.get("currency")
    titular = result.get("titular")
    merchant = result.get("merchant")

    if currency == "USD":
        return "en dólares"
    if currency == "ARS":
        return "en pesos"
    if titular:
        return f"de {titular}"
    if merchant:
        return f"para {merchant}"
    return ""


def _count_suffix(result: dict[str, Any]) -> str:
    currency = result.get("currency")
    if currency == "USD":
        return " en USD"
    if currency == "ARS":
        return " en pesos"
    if result.get("titular"):
        return f" de {result['titular']}"
    return ""
