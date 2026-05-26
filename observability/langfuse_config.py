import os
from typing import Any

from langfuse import Langfuse, get_client

from observability.metrics import is_langfuse_observability_enabled


def get_langfuse_handler():
    if not is_langfuse_observability_enabled():
        print("[observability] Langfuse observability disabled")
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print("[observability] Langfuse not configured")
        return None

    try:
        from langfuse.langchain import CallbackHandler

        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

        langfuse = get_client()

        if not langfuse.auth_check():
            print("[observability] Langfuse auth_check failed")
            return None

        print("[observability] Langfuse connected")
        return CallbackHandler()
    except Exception:
        print("[observability] Langfuse initialization failed")
        return None


def safe_update_observation(
    observation: Any,
    *,
    metadata: dict[str, Any] | None = None,
    level: str | None = None,
    status_message: str | None = None,
    version: str | None = None,
) -> bool:
    if observation is None:
        return False

    try:
        kwargs: dict[str, Any] = {}

        if metadata is not None:
            kwargs["metadata"] = metadata
        if level:
            kwargs["level"] = level
        if status_message:
            kwargs["status_message"] = status_message
        if version:
            kwargs["version"] = version

        if not kwargs:
            return False

        observation.update(**kwargs)
        return True
    except Exception:
        return False


def safe_score(
    langfuse_target: Any,
    trace_id: str | None,
    name: str,
    value: int | float | str | bool,
    *,
    data_type: str | None = None,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
    scope: str = "trace",
) -> bool:
    if langfuse_target is None or not name:
        return False

    kwargs: dict[str, Any] = {
        "name": name,
        "value": value,
    }

    if data_type:
        kwargs["data_type"] = data_type
    if comment:
        kwargs["comment"] = comment
    if metadata is not None:
        kwargs["metadata"] = metadata

    try:
        if scope == "observation" and hasattr(langfuse_target, "score"):
            langfuse_target.score(**kwargs)
            return True

        if hasattr(langfuse_target, "score_trace"):
            langfuse_target.score_trace(**kwargs)
            return True

        if hasattr(langfuse_target, "create_score") and trace_id:
            langfuse_target.create_score(trace_id=trace_id, **kwargs)
            return True

        if hasattr(langfuse_target, "score_current_trace"):
            langfuse_target.score_current_trace(**kwargs)
            return True
    except Exception:
        return False

    return False


def safe_numeric_score(
    langfuse_target: Any,
    trace_id: str | None,
    name: str,
    value: Any,
    *,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
    scope: str = "trace",
) -> bool:
    try:
        if value is None or value == "":
            return False
        numeric_value = float(value)
        if numeric_value.is_integer():
            numeric_value = int(numeric_value)
    except (TypeError, ValueError):
        return False

    return safe_score(
        langfuse_target,
        trace_id,
        name,
        numeric_value,
        data_type="NUMERIC",
        comment=comment,
        metadata=metadata,
        scope=scope,
    )


def safe_boolean_score(
    langfuse_target: Any,
    trace_id: str | None,
    name: str,
    value: Any,
    *,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
    scope: str = "trace",
) -> bool:
    normalized_value = _normalize_bool_score_value(value)
    if normalized_value is None:
        return False

    return safe_score(
        langfuse_target,
        trace_id,
        name,
        normalized_value,
        data_type="BOOLEAN",
        comment=comment,
        metadata=metadata,
        scope=scope,
    )


def safe_categorical_score(
    langfuse_target: Any,
    trace_id: str | None,
    name: str,
    value: Any,
    *,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
    scope: str = "trace",
) -> bool:
    normalized_value = str(value or "").strip()
    if not normalized_value:
        return False

    return safe_score(
        langfuse_target,
        trace_id,
        name,
        normalized_value,
        data_type="CATEGORICAL",
        comment=comment,
        metadata=metadata,
        scope=scope,
    )


def _normalize_bool_score_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0

    if isinstance(value, (int, float)):
        return 1 if value else 0

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return 1
        if lowered in {"0", "false", "no", "off"}:
            return 0

    return None
