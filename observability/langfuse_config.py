import os
from typing import Any

from langfuse import Langfuse, get_client as get_langfuse_sdk_client, propagate_attributes

from observability.metrics import is_langfuse_observability_enabled

_MAX_PROPAGATED_ATTRIBUTE_LENGTH = 200


def get_langfuse_client():
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
        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

        langfuse = get_langfuse_sdk_client()

        if not langfuse.auth_check():
            print("[observability] Langfuse auth_check failed")
            return None

        print("[observability] Langfuse connected")
        return langfuse
    except Exception:
        print("[observability] Langfuse initialization failed")
        return None


def safe_update_current_trace(
    langfuse_client: Any,
    *,
    name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    if langfuse_client is None:
        return False

    normalized_name = _normalize_trace_string(name)
    normalized_user_id = _normalize_trace_string(user_id)
    normalized_session_id = _normalize_trace_string(session_id)
    normalized_tags = _normalize_trace_tags(tags)
    normalized_trace_metadata = _normalize_trace_metadata(metadata)

    direct_update_kwargs: dict[str, Any] = {}
    if normalized_name:
        direct_update_kwargs["name"] = normalized_name
    if normalized_user_id:
        direct_update_kwargs["user_id"] = normalized_user_id
    if normalized_session_id:
        direct_update_kwargs["session_id"] = normalized_session_id
    if normalized_tags:
        direct_update_kwargs["tags"] = normalized_tags
    if metadata:
        direct_update_kwargs["metadata"] = metadata

    if direct_update_kwargs:
        for method_name in ("update_current_trace", "update_trace"):
            update_trace = getattr(langfuse_client, method_name, None)
            if not callable(update_trace):
                continue

            try:
                update_trace(**direct_update_kwargs)
                return True
            except Exception:
                return False

    propagation_kwargs: dict[str, Any] = {}
    if normalized_name:
        propagation_kwargs["trace_name"] = normalized_name
    if normalized_user_id:
        propagation_kwargs["user_id"] = normalized_user_id
    if normalized_session_id:
        propagation_kwargs["session_id"] = normalized_session_id
    if normalized_tags:
        propagation_kwargs["tags"] = normalized_tags
    if normalized_trace_metadata:
        propagation_kwargs["metadata"] = normalized_trace_metadata

    if not propagation_kwargs:
        return False

    try:
        with propagate_attributes(**propagation_kwargs):
            pass
        return True
    except Exception:
        return False


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


def _normalize_trace_string(value: Any) -> str | None:
    if value is None:
        return None

    normalized_value = str(value).strip()
    if not normalized_value:
        return None

    return normalized_value[:_MAX_PROPAGATED_ATTRIBUTE_LENGTH]


def _normalize_trace_tags(tags: list[str] | tuple[str, ...] | None) -> list[str] | None:
    if not isinstance(tags, (list, tuple)):
        return None

    normalized_tags: list[str] = []
    for tag in tags:
        normalized_tag = _normalize_trace_string(tag)
        if not normalized_tag or normalized_tag in normalized_tags:
            continue
        normalized_tags.append(normalized_tag)

    return normalized_tags or None


def _normalize_trace_metadata(metadata: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(metadata, dict):
        return None

    normalized_metadata: dict[str, str] = {}
    for raw_key, raw_value in metadata.items():
        normalized_key = _normalize_trace_string(raw_key)
        normalized_value = _normalize_trace_metadata_value(raw_value)
        if not normalized_key or normalized_value is None:
            continue
        normalized_metadata[normalized_key] = normalized_value

    return normalized_metadata or None


def _normalize_trace_metadata_value(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return "true" if value else "false"

    normalized_value = str(value).strip()
    if not normalized_value:
        return None

    return normalized_value[:_MAX_PROPAGATED_ATTRIBUTE_LENGTH]


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
