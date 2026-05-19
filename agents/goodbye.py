from __future__ import annotations

import os
import re
import unicodedata

from agents.state import AgentState
from observability.logger import log_step


DEFAULT_CSAT_FLOW_TEMPLATE_SID = "HX6e85fd5396aeef5fed1018ceb7c69dbd"
CSAT_FLOW_TEMPLATE_SID = (
    os.getenv("CSAT_FLOW_TEMPLATE_SID", DEFAULT_CSAT_FLOW_TEMPLATE_SID).strip()
    or DEFAULT_CSAT_FLOW_TEMPLATE_SID
)

GOODBYE_ANSWER = (
    "\u00a1Gracias por escribirme! \U0001F60A Antes de irte, te dejo una encuesta "
    "s\u00faper r\u00e1pida para saber c\u00f3mo fue tu experiencia."
)
GOODBYE_REPEAT_ANSWER = "\u00a1Gracias por escribirme! \U0001F60A"

GOODBYE_PHRASES = (
    "chau",
    "adios",
    "hasta luego",
    "nos vemos",
)

GOODBYE_FILLER_TOKENS = {
    "chau",
    "adios",
    "hasta",
    "luego",
    "nos",
    "vemos",
    "gracias",
    "muchas",
    "mil",
    "bueno",
    "dale",
    "listo",
    "ok",
    "oka",
    "perfecto",
    "genial",
    "gala",
    "che",
}

GOODBYE_NEGATIVE_PATTERNS = (
    "que significa chau",
    "que significa adios",
    "si digo chau",
    "si digo adios",
    "quiero configurar el flujo de chau",
    "quiero configurar el flujo de adios",
    "usuario dijo chau",
    "usuario dijo adios",
    "en el ejemplo",
    "flujo de chau",
    "flujo de adios",
)


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


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def is_goodbye_message(text: str) -> bool:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return False

    if any(pattern in normalized_text for pattern in GOODBYE_NEGATIVE_PATTERNS):
        return False

    if not any(_contains_phrase(normalized_text, phrase) for phrase in GOODBYE_PHRASES):
        return False

    tokens = normalized_text.split()
    if not tokens or len(tokens) > 5:
        return False

    return all(token in GOODBYE_FILLER_TOKENS for token in tokens)


def goodbye_node(state: AgentState) -> AgentState:
    memory = state.get("memory") or {}
    csat_already_sent = bool(memory.get("csat_sent"))

    if csat_already_sent:
        answer = GOODBYE_REPEAT_ANSWER
        send_csat = False
    else:
        answer = GOODBYE_ANSWER
        send_csat = True

    log_step(
        "GOODBYE",
        "Cierre conversacional generado",
        {
            "send_csat": send_csat,
            "csat_already_sent": csat_already_sent,
        },
    )

    return {
        **state,
        "answer": answer,
        "topic": "despedida",
        "send_csat": send_csat,
        "csat_template_sid": CSAT_FLOW_TEMPLATE_SID,
        "error": None,
    }
