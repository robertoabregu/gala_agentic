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
    value: int | float | str,
    *,
    data_type: str | None = None,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
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
