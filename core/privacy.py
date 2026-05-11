import re
from typing import Any


IDENTIFICATION_PATTERN = re.compile(r"(?<!\d)(?:\d[\s-]?){11}(?!\d)")


def normalize_identification(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def extract_identification(text: str) -> str | None:
    if not text:
        return None

    for match in IDENTIFICATION_PATTERN.finditer(text):
        digits = normalize_identification(match.group(0))
        if len(digits) == 11:
            return digits

    return None


def mask_identification(value: str) -> str:
    digits = normalize_identification(value)

    if not digits:
        return ""

    if len(digits) <= 4:
        return "*" * len(digits)

    return f"{digits[:2]}*******{digits[-2:]}"


def mask_sensitive_text(text: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        digits = normalize_identification(match.group(0))
        if len(digits) != 11:
            return match.group(0)
        return mask_identification(digits)

    return IDENTIFICATION_PATTERN.sub(replace, text)


def sanitize_for_logging(value: Any) -> Any:
    if isinstance(value, str):
        return mask_sensitive_text(value)

    if isinstance(value, int):
        digits = str(value)
        if len(digits) == 11:
            return mask_identification(digits)
        return value

    if isinstance(value, list):
        return [sanitize_for_logging(item) for item in value]

    if isinstance(value, tuple):
        return tuple(sanitize_for_logging(item) for item in value)

    if isinstance(value, dict):
        return {
            key: sanitize_for_logging(item)
            for key, item in value.items()
        }

    return value
