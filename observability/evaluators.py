import json
import os
import re
from typing import Any

from openai import OpenAI

from core.privacy import mask_sensitive_text
from observability.metrics import build_basic_scores, bool_score, safe_len


FALSE_ENV_VALUES = {"0", "false", "no", "off"}
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_MAX_INPUT_CHARS = 1000
DEFAULT_MAX_CONTEXT_CHARS = 2000
DEFAULT_WHATSAPP_MAX_CHARS = 1800
DEFAULT_MAX_EMOJIS = 6
DEFAULT_MAX_NEWLINES = 14
DEFAULT_MAX_BULLET_CHARS = 220
BANKING_ROUTES = {
    "loans_rag",
    "bcra_credit_status",
    "branch_locator",
    "benefits",
    "credit_card_statement",
    "fallback",
    "sensitive",
}
QUALITY_SCORE_DEFINITIONS = {
    "whatsapp_format_ok": {
        "data_type": "BOOLEAN",
        "comment": "Indica si la respuesta parece compatible con WhatsApp.",
    },
    "whatsapp_format_issues_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad de problemas detectados de formato WhatsApp.",
    },
    "whatsapp_has_broken_bold": {
        "data_type": "BOOLEAN",
        "comment": "Indica si hay negritas con asteriscos impares o rotos.",
    },
    "whatsapp_has_double_asterisk": {
        "data_type": "BOOLEAN",
        "comment": "Indica si se uso markdown web con doble asterisco.",
    },
    "whatsapp_answer_too_long": {
        "data_type": "BOOLEAN",
        "comment": "Indica si la respuesta parece demasiado larga para WhatsApp.",
    },
    "likely_low_value_answer": {
        "data_type": "BOOLEAN",
        "comment": "Indica si la respuesta parece de bajo valor o poco util.",
    },
    "routing_correctness": {
        "data_type": "NUMERIC",
        "comment": "Evaluacion de correccion de ruta entre 0 y 1.",
    },
    "answer_relevance": {
        "data_type": "NUMERIC",
        "comment": "Evaluacion de relevancia de la respuesta entre 0 y 1.",
    },
    "groundedness": {
        "data_type": "NUMERIC",
        "comment": "Evaluacion de fundamentacion en contexto entre 0 y 1.",
    },
    "hallucination_risk": {
        "data_type": "NUMERIC",
        "comment": "Riesgo estimado de alucinacion entre 0 y 1.",
    },
    "helpfulness": {
        "data_type": "NUMERIC",
        "comment": "Evaluacion de utilidad de la respuesta entre 0 y 1.",
    },
    "needs_human_review": {
        "data_type": "BOOLEAN",
        "comment": "Indica si la conversacion deberia revisarse manualmente.",
    },
    "dataset_candidate": {
        "data_type": "BOOLEAN",
        "comment": "Indica si la conversacion es candidata para dataset o regresion.",
    },
}


def is_langfuse_quality_eval_enabled() -> bool:
    raw_value = os.getenv("LANGFUSE_QUALITY_EVAL_ENABLED")
    if raw_value is None:
        return True

    return raw_value.strip().lower() not in FALSE_ENV_VALUES


def is_llm_judge_enabled() -> bool:
    raw_value = os.getenv("LLM_JUDGE_ENABLED")
    if raw_value is None:
        return False

    return raw_value.strip().lower() not in FALSE_ENV_VALUES


def get_judge_model() -> str:
    return (
        os.getenv("OPENAI_MODEL_JUDGE", "").strip()
        or os.getenv("OPENAI_MODEL_FAST", "").strip()
        or os.getenv("OPENAI_CHAT_MODEL", "").strip()
        or DEFAULT_JUDGE_MODEL
    )


def evaluate_whatsapp_format(answer: str) -> dict[str, int]:
    text = str(answer or "")
    lines = text.splitlines()
    emoji_count = _count_emojis(text)
    bullet_lines = [
        line
        for line in lines
        if line.lstrip().startswith(("-", "*", "•"))
    ]

    has_broken_bold = int(text.count("*") % 2 != 0)
    has_double_asterisk = int("**" in text)
    answer_too_long = int(safe_len(text) > DEFAULT_WHATSAPP_MAX_CHARS)
    has_code_block = int("```" in text)
    has_excessive_newlines = int(text.count("\n") > DEFAULT_MAX_NEWLINES or "\n\n\n" in text)
    has_too_many_emojis = int(emoji_count > DEFAULT_MAX_EMOJIS)
    has_long_bullets = int(
        any(safe_len(line.strip()) > DEFAULT_MAX_BULLET_CHARS for line in bullet_lines)
    )
    is_empty = int(not text.strip())

    issues_count = sum(
        [
            is_empty,
            has_broken_bold,
            has_double_asterisk,
            answer_too_long,
            has_code_block,
            has_excessive_newlines,
            has_too_many_emojis,
            has_long_bullets,
        ]
    )

    return {
        "whatsapp_format_ok": 1 if issues_count == 0 else 0,
        "whatsapp_format_issues_count": issues_count,
        "whatsapp_has_broken_bold": has_broken_bold,
        "whatsapp_has_double_asterisk": has_double_asterisk,
        "whatsapp_answer_too_long": answer_too_long,
        "whatsapp_has_code_block": has_code_block,
        "whatsapp_has_excessive_newlines": has_excessive_newlines,
        "whatsapp_has_too_many_emojis": has_too_many_emojis,
        "whatsapp_has_long_bullets": has_long_bullets,
    }


def evaluate_basic_quality_signals(state: dict[str, Any]) -> dict[str, int]:
    safe_state = state if isinstance(state, dict) else {}
    base_scores = build_basic_scores(safe_state)
    question = _safe_text(
        safe_state.get("question")
        or safe_state.get("standalone_question")
        or safe_state.get("original_question")
    )
    answer = _final_answer(safe_state)
    route = _route(safe_state)
    tool_output = _tool_output(safe_state)
    answer_empty = int(not answer.strip())
    answer_length = safe_len(answer)
    used_rag = base_scores["used_rag"]
    used_tool = base_scores["used_tool"]
    has_context = base_scores["has_context"]
    retrieval_docs_count = base_scores["retrieval_docs_count"]
    fallback_used = base_scores["fallback_used"]
    guardrail_blocked = base_scores["guardrail_blocked"]
    needs_clarification = base_scores["needs_clarification"]
    tool_error = int(
        used_tool
        and bool(
            _safe_text(safe_state.get("error"))
            or _safe_text(tool_output.get("error"))
            or _safe_text(tool_output.get("error_type"))
        )
    )
    question_complex = int(_is_complex_question(question, route))
    too_short_for_complex_question = int(question_complex and answer_length < 24)

    likely_low_value_answer = int(
        answer_empty
        or fallback_used
        or too_short_for_complex_question
        or (used_rag and retrieval_docs_count == 0)
        or (used_rag and not has_context)
        or tool_error
    )

    return {
        "answer_empty": answer_empty,
        "answer_length": answer_length,
        "fallback_used": fallback_used,
        "used_rag": used_rag,
        "used_tool": used_tool,
        "has_context": has_context,
        "retrieval_docs_count": retrieval_docs_count,
        "guardrail_blocked": guardrail_blocked,
        "needs_clarification": needs_clarification,
        "tool_error": tool_error,
        "question_complex": question_complex,
        "too_short_for_complex_question": too_short_for_complex_question,
        "likely_low_value_answer": likely_low_value_answer,
    }


def should_run_llm_judge(state: dict[str, Any]) -> bool:
    safe_state = state if isinstance(state, dict) else {}

    if not is_langfuse_quality_eval_enabled():
        return False
    if not is_llm_judge_enabled():
        return False
    if not _final_answer(safe_state).strip():
        return False

    return True


def run_llm_quality_judge(
    state: dict[str, Any],
    client: OpenAI | None = None,
) -> dict[str, Any]:
    if not should_run_llm_judge(state):
        return {"skipped": True}

    openai_client = client or _build_openai_client_from_env()
    if openai_client is None:
        return {"skipped": True, "error": "judge_client_unavailable"}

    judge_model = get_judge_model()
    question_max_chars = _int_env("LLM_JUDGE_MAX_INPUT_CHARS", DEFAULT_MAX_INPUT_CHARS, 200)
    context_max_chars = _int_env("LLM_JUDGE_MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS, 400)
    payload = _build_llm_judge_payload(
        state,
        question_max_chars=question_max_chars,
        context_max_chars=context_max_chars,
    )

    response = openai_client.chat.completions.create(
        model=judge_model,
        temperature=0,
        response_format={"type": "json_object"},
        timeout=15,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sos un evaluador de calidad para un chatbot bancario. "
                    "Debes evaluar la conversacion y responder SOLO con JSON estricto. "
                    "No cambies la respuesta, no des sugerencias conversacionales y no incluyas texto fuera del JSON.\n\n"
                    "Escalas:\n"
                    "- routing_correctness: 0 a 1\n"
                    "- answer_relevance: 0 a 1\n"
                    "- groundedness: 0 a 1\n"
                    "- hallucination_risk: 0 a 1, donde 1 es mayor riesgo\n"
                    "- helpfulness: 0 a 1\n"
                    "- needs_human_review: 0 o 1\n"
                    "- dataset_candidate: 0 o 1\n\n"
                    "Evalua usando solo la informacion provista. "
                    "Si no hay contexto RAG, groundedness puede ser 1 cuando la respuesta no depende de contexto externo. "
                    "La razon debe ser breve, sin repetir datos sensibles ni texto completo del usuario.\n\n"
                    "Devuelve exactamente este formato:\n"
                    "{"
                    "\"routing_correctness\": 0.0, "
                    "\"answer_relevance\": 0.0, "
                    "\"groundedness\": 0.0, "
                    "\"hallucination_risk\": 0.0, "
                    "\"helpfulness\": 0.0, "
                    "\"needs_human_review\": 0, "
                    "\"dataset_candidate\": 0, "
                    "\"reason\": \"explicacion breve\""
                    "}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=True),
            },
        ],
    )

    raw_content = response.choices[0].message.content if response.choices else "{}"
    parsed = _parse_judge_json(raw_content)
    return {
        "routing_correctness": _clamp_float(parsed.get("routing_correctness")),
        "answer_relevance": _clamp_float(parsed.get("answer_relevance")),
        "groundedness": _clamp_float(parsed.get("groundedness"), default=1.0),
        "hallucination_risk": _clamp_float(parsed.get("hallucination_risk")),
        "helpfulness": _clamp_float(parsed.get("helpfulness")),
        "needs_human_review": _clamp_bool(parsed.get("needs_human_review")),
        "dataset_candidate": _clamp_bool(parsed.get("dataset_candidate")),
        "reason": _short_reason(parsed.get("reason")),
        "judge_model": judge_model,
        "skipped": False,
    }


def build_quality_scores(
    state: dict[str, Any],
    *,
    whatsapp_eval: dict[str, int] | None = None,
    basic_signals: dict[str, int] | None = None,
    llm_judge_result: dict[str, Any] | None = None,
) -> dict[str, int | float]:
    safe_state = state if isinstance(state, dict) else {}
    whatsapp = whatsapp_eval or evaluate_whatsapp_format(_final_answer(safe_state))
    basic = basic_signals or evaluate_basic_quality_signals(safe_state)
    llm = llm_judge_result or {}

    fallback_used = basic.get("fallback_used", 0)
    guardrail_blocked = basic.get("guardrail_blocked", 0)
    answer_empty = basic.get("answer_empty", 0)
    too_short_for_complex_question = basic.get("too_short_for_complex_question", 0)
    route = _route(safe_state)
    used_rag = basic.get("used_rag", 0)
    llm_enabled = bool(llm) and not llm.get("skipped")

    routing_correctness = _clamp_float(llm.get("routing_correctness"), default=1.0 if route else 0.0)
    answer_relevance = _clamp_float(
        llm.get("answer_relevance"),
        default=0.2 if fallback_used or answer_empty else 0.7,
    )
    groundedness = _clamp_float(
        llm.get("groundedness"),
        default=0.3 if used_rag and not basic.get("has_context") else 0.8,
    )
    hallucination_risk = _clamp_float(
        llm.get("hallucination_risk"),
        default=0.75 if fallback_used and route in BANKING_ROUTES else 0.25,
    )
    helpfulness = _clamp_float(
        llm.get("helpfulness"),
        default=0.2 if basic.get("likely_low_value_answer") else 0.75,
    )

    heuristic_human_review = int(
        basic.get("likely_low_value_answer", 0)
        or guardrail_blocked
        or _safe_text(safe_state.get("error"))
        or (fallback_used and route in BANKING_ROUTES)
        or (whatsapp.get("whatsapp_format_ok", 1) == 0)
        or answer_empty
        or too_short_for_complex_question
    )
    if llm_enabled:
        heuristic_human_review = int(
            heuristic_human_review
            or routing_correctness < 0.5
            or answer_relevance < 0.45
            or hallucination_risk > 0.65
            or (used_rag and groundedness < 0.45)
        )

    heuristic_dataset_candidate = int(
        heuristic_human_review
        or fallback_used
        or basic.get("likely_low_value_answer", 0)
        or (whatsapp.get("whatsapp_format_ok", 1) == 0)
    )
    if llm_enabled:
        heuristic_dataset_candidate = int(
            heuristic_dataset_candidate
            or routing_correctness < 0.75
            or answer_relevance < 0.6
            or groundedness < 0.6
            or hallucination_risk > 0.5
        )

    needs_human_review = max(
        heuristic_human_review,
        _clamp_bool(llm.get("needs_human_review")),
    )
    dataset_candidate = max(
        heuristic_dataset_candidate,
        _clamp_bool(llm.get("dataset_candidate")),
    )

    scores: dict[str, int | float] = {
        "whatsapp_format_ok": whatsapp.get("whatsapp_format_ok", 1),
        "whatsapp_format_issues_count": whatsapp.get("whatsapp_format_issues_count", 0),
        "whatsapp_has_broken_bold": whatsapp.get("whatsapp_has_broken_bold", 0),
        "whatsapp_has_double_asterisk": whatsapp.get("whatsapp_has_double_asterisk", 0),
        "whatsapp_answer_too_long": whatsapp.get("whatsapp_answer_too_long", 0),
        "likely_low_value_answer": basic.get("likely_low_value_answer", 0),
        "needs_human_review": needs_human_review,
        "dataset_candidate": dataset_candidate,
    }

    if llm_enabled:
        scores.update(
            {
                "routing_correctness": routing_correctness,
                "answer_relevance": answer_relevance,
                "groundedness": groundedness,
                "hallucination_risk": hallucination_risk,
                "helpfulness": helpfulness,
            }
        )

    return scores


def build_quality_metadata(
    state: dict[str, Any],
    *,
    whatsapp_eval: dict[str, int] | None = None,
    basic_signals: dict[str, int] | None = None,
    llm_judge_result: dict[str, Any] | None = None,
    quality_scores: dict[str, int | float] | None = None,
    quality_eval_error: str | None = None,
) -> dict[str, Any]:
    safe_state = state if isinstance(state, dict) else {}
    whatsapp = whatsapp_eval or evaluate_whatsapp_format(_final_answer(safe_state))
    basic = basic_signals or evaluate_basic_quality_signals(safe_state)
    llm = llm_judge_result or {}
    scores = quality_scores or build_quality_scores(
        safe_state,
        whatsapp_eval=whatsapp,
        basic_signals=basic,
        llm_judge_result=llm,
    )
    llm_enabled = is_llm_judge_enabled()
    judge_model = get_judge_model() if llm_enabled else ""

    return {
        "quality_eval_enabled": is_langfuse_quality_eval_enabled(),
        "llm_judge_enabled": llm_enabled,
        "judge_model": judge_model,
        "needs_human_review": int(scores.get("needs_human_review", 0)),
        "dataset_candidate": int(scores.get("dataset_candidate", 0)),
        "quality_eval_error": _short_reason(quality_eval_error or llm.get("error")),
        "quality_reason_short": _short_reason(
            llm.get("reason")
            or _build_programmatic_quality_reason(whatsapp, basic, scores)
        ),
    }


def run_quality_evaluation(
    state: dict[str, Any],
    *,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    metadata = {
        "quality_eval_enabled": is_langfuse_quality_eval_enabled(),
        "llm_judge_enabled": is_llm_judge_enabled(),
        "judge_model": get_judge_model() if is_llm_judge_enabled() else "",
        "needs_human_review": 0,
        "dataset_candidate": 0,
        "quality_eval_error": None,
        "quality_reason_short": "",
    }
    if not is_langfuse_quality_eval_enabled():
        return {
            "scores": {},
            "metadata": metadata,
            "whatsapp_eval": {},
            "basic_signals": {},
            "llm_judge_result": {"skipped": True},
            "llm_judge_ran": False,
        }

    whatsapp_eval = evaluate_whatsapp_format(_final_answer(state))
    basic_signals = evaluate_basic_quality_signals(state)
    llm_judge_result: dict[str, Any] = {"skipped": True}
    quality_eval_error = None
    llm_judge_ran = False

    if should_run_llm_judge(state):
        try:
            llm_judge_result = run_llm_quality_judge(state, client=client)
            llm_judge_ran = not llm_judge_result.get("skipped", False)
            if llm_judge_result.get("error"):
                quality_eval_error = _short_reason(llm_judge_result.get("error"))
        except Exception as exc:
            quality_eval_error = _short_reason(f"{type(exc).__name__}: {str(exc)}")
            llm_judge_result = {
                "skipped": True,
                "error": quality_eval_error,
            }

    scores = build_quality_scores(
        state,
        whatsapp_eval=whatsapp_eval,
        basic_signals=basic_signals,
        llm_judge_result=llm_judge_result,
    )
    metadata = build_quality_metadata(
        state,
        whatsapp_eval=whatsapp_eval,
        basic_signals=basic_signals,
        llm_judge_result=llm_judge_result,
        quality_scores=scores,
        quality_eval_error=quality_eval_error,
    )

    return {
        "scores": scores,
        "metadata": metadata,
        "whatsapp_eval": whatsapp_eval,
        "basic_signals": basic_signals,
        "llm_judge_result": llm_judge_result,
        "llm_judge_ran": llm_judge_ran,
    }


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _final_answer(state: dict[str, Any]) -> str:
    return _safe_text(state.get("final_answer")) or _safe_text(state.get("answer"))


def _route(state: dict[str, Any]) -> str:
    route = _safe_text(state.get("route"))
    if route:
        return route

    memory = state.get("memory") or {}
    if isinstance(memory, dict):
        return _safe_text(memory.get("last_route"))

    return ""


def _tool_output(state: dict[str, Any]) -> dict[str, Any]:
    tool_output = state.get("tool_output")
    if isinstance(tool_output, dict):
        return tool_output
    return {}


def _is_complex_question(question: str, route: str) -> bool:
    normalized = question.lower()
    if route in BANKING_ROUTES and route != "fallback":
        return True
    if safe_len(question.split()) >= 6:
        return True
    return any(
        token in normalized
        for token in (
            "como",
            "cuanto",
            "cuáles",
            "cuales",
            "requisitos",
            "prestamo",
            "préstamo",
            "beneficios",
            "deuda",
            "tarjeta",
        )
    )


def _count_emojis(text: str) -> int:
    return len(
        re.findall(
            (
                "["
                "\U0001F300-\U0001F5FF"
                "\U0001F600-\U0001F64F"
                "\U0001F680-\U0001F6FF"
                "\U0001F700-\U0001F77F"
                "\U0001F900-\U0001F9FF"
                "\U0001FA70-\U0001FAFF"
                "]+"
            ),
            text or "",
        )
    )


def _int_env(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        return max(minimum, int(raw_value or default))
    except (TypeError, ValueError):
        return default


def _build_openai_client_from_env() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _build_llm_judge_payload(
    state: dict[str, Any],
    *,
    question_max_chars: int,
    context_max_chars: int,
) -> dict[str, Any]:
    question = _limit_text(
        mask_sensitive_text(
            _safe_text(
                state.get("question")
                or state.get("standalone_question")
                or state.get("original_question")
            )
        ),
        question_max_chars,
    )
    answer = _limit_text(mask_sensitive_text(_final_answer(state)), question_max_chars)
    context = _limit_text(
        mask_sensitive_text(_safe_text(state.get("context"))),
        context_max_chars,
    )

    return {
        "user_question": question,
        "final_answer": answer,
        "selected_route": _route(state),
        "final_topic": _safe_text(state.get("topic")),
        "retrieved_context_snippet": context,
        "documents_count": safe_len(state.get("documents") or []),
        "used_rag": bool(build_basic_scores(state).get("used_rag", 0)),
        "used_tool": bool(build_basic_scores(state).get("used_tool", 0)),
        "fallback": bool(build_basic_scores(state).get("fallback_used", 0)),
        "guardrail_blocked": bool(build_basic_scores(state).get("guardrail_blocked", 0)),
    }


def _limit_text(text: str, max_chars: int) -> str:
    cleaned = _safe_text(text)
    if safe_len(cleaned) <= max_chars:
        return cleaned

    return f"{cleaned[:max_chars].rstrip()}..."


def _parse_judge_json(raw_content: Any) -> dict[str, Any]:
    if isinstance(raw_content, list):
        joined_parts = []
        for part in raw_content:
            if isinstance(part, dict) and "text" in part:
                joined_parts.append(str(part.get("text") or ""))
            else:
                joined_parts.append(str(part))
        raw_text = "".join(joined_parts)
    else:
        raw_text = str(raw_content or "")

    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE)

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start >= 0 and end > start:
        raw_text = raw_text[start:end + 1]

    parsed = json.loads(raw_text or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("LLM judge did not return a JSON object.")
    return parsed


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default

    return max(0.0, min(1.0, numeric))


def _clamp_bool(value: Any) -> int:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return 1
        if lowered in {"0", "false", "no"}:
            return 0

    return bool_score(value)


def _short_reason(reason: Any, max_chars: int = 160) -> str | None:
    text = _safe_text(mask_sensitive_text(str(reason or "")))
    if not text:
        return None

    if safe_len(text) <= max_chars:
        return text

    return f"{text[:max_chars].rstrip()}..."


def _build_programmatic_quality_reason(
    whatsapp_eval: dict[str, int],
    basic_signals: dict[str, int],
    scores: dict[str, int | float],
) -> str:
    reasons: list[str] = []

    if basic_signals.get("likely_low_value_answer"):
        reasons.append("low_value_signal")
    if basic_signals.get("fallback_used"):
        reasons.append("fallback_used")
    if basic_signals.get("guardrail_blocked"):
        reasons.append("guardrail_blocked")
    if whatsapp_eval.get("whatsapp_format_ok", 1) == 0:
        reasons.append("whatsapp_format")
    if scores.get("needs_human_review"):
        reasons.append("needs_review")

    if not reasons:
        reasons.append("quality_ok")

    return ", ".join(reasons[:4])
