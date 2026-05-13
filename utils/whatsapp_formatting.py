from __future__ import annotations

import re
import unicodedata


KNOWN_EMOJIS = (
    "💰",
    "🏠",
    "🚗",
    "📌",
    "ℹ️",
    "📄",
    "📱",
    "📍",
    "📊",
    "👋",
    "😊",
    "🚶",
)

LOANS_ROUTES = {"loans_rag", "rag"}
LOANS_TOPICS = {"prestamo", "prestamos", "loan", "loans"}
EMOJI_BLOCKED_ROUTES = {
    "bcra_credit_status",
    "branch_locator",
    "chitchat",
    "fallback",
    "sensitive",
}
EMOJI_BLOCKED_TOPICS = {
    "conversacion",
    "fallback",
    "situacion_crediticia_bcra",
    "sensitive",
    "sucursales_cercanas",
}

MARKDOWN_LINK_PATTERN = re.compile(
    r"\[([^\]\n]+)\]\(((?:https?://|www\.)[^\s)]+)\)",
    flags=re.IGNORECASE,
)
URL_PATTERN = re.compile(
    r"(?P<url>(?:https?://|www\.)[^\s<>()]+)",
    flags=re.IGNORECASE,
)
HEADER_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
STAR_BULLET_PATTERN = re.compile(r"^(?P<indent>\s*)\*\s+(?P<item>\S.*)$")
TRIPLE_ASTERISK_PATTERN = re.compile(r"\*\*\*([^\n*][^\n]*?[^\n*])\*\*\*")
DOUBLE_ASTERISK_PATTERN = re.compile(r"\*\*([^\n*][^\n]*?[^\n*])\*\*")
DOUBLE_UNDERSCORE_PATTERN = re.compile(r"__([^\n_][^\n]*?[^\n_])__")
BROKEN_DOUBLE_OPEN_PATTERN = re.compile(r"\*\*([^\n*]+?)\*")
BROKEN_DOUBLE_CLOSE_PATTERN = re.compile(r"\*([^\n*]+?)\*\*")
BROKEN_UNDERSCORE_OPEN_PATTERN = re.compile(r"__([^\n_]+?)_")
BROKEN_UNDERSCORE_CLOSE_PATTERN = re.compile(r"_([^\n_]+?)__")
VALID_WHATSAPP_BOLD_PATTERN = re.compile(r"\*(?=\S)([^*\n]+?)(?<=\S)\*")
EMOJI_PREFIX_ALTERNATION = "|".join(re.escape(emoji) for emoji in KNOWN_EMOJIS)
LABEL_COLON_OUTSIDE_PATTERN = re.compile(
    rf"^(?P<indent>\s*)(?P<emoji>(?:{EMOJI_PREFIX_ALTERNATION})\s+)?"
    r"\*(?P<label>[^*\n]{1,60}?)\*\s*:\s*(?P<rest>.*)$"
)
LABEL_COLON_INSIDE_PATTERN = re.compile(
    rf"^(?P<indent>\s*)(?P<emoji>(?:{EMOJI_PREFIX_ALTERNATION})\s+)?"
    r"\*(?P<label>[^*\n:]{1,60}?)\:\*\s*(?P<rest>.*)$"
)
EMOJI_REGEX = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\u2139]")


def format_whatsapp_answer(
    text: str,
    *,
    topic: str | None = None,
    route: str | None = None,
) -> str:
    if not text:
        return text

    formatted = text.replace("\r\n", "\n").strip()
    formatted = _sanitize_unsupported_markdown(formatted)
    formatted = _normalize_whatsapp_bold(formatted)
    formatted = _normalize_label_colon_format(
        formatted,
        route=route,
        topic=topic,
    )
    formatted = _add_discreet_emojis(
        formatted,
        route=route,
        topic=topic,
    )
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)
    return formatted.strip()


def _sanitize_unsupported_markdown(text: str) -> str:
    text = MARKDOWN_LINK_PATTERN.sub(_replace_markdown_link, text)
    protected_text, url_tokens = _protect_urls(text)

    sanitized_lines: list[str] = []

    for line in protected_text.split("\n"):
        header_match = HEADER_PATTERN.match(line)
        if header_match:
            title = header_match.group(1).strip().strip("#").strip()
            sanitized_lines.append(f"*{title}*" if title else "")
            continue

        bullet_match = STAR_BULLET_PATTERN.match(line)
        if bullet_match:
            sanitized_lines.append(
                f"{bullet_match.group('indent')}• {bullet_match.group('item').strip()}"
            )
            continue

        sanitized_lines.append(re.sub(r"`([^`\n]+)`", r"\1", line))

    return _restore_tokens("\n".join(sanitized_lines), url_tokens, prefix="URLTOKEN")


def _normalize_whatsapp_bold(text: str) -> str:
    protected_text, url_tokens = _protect_urls(text)

    protected_text = TRIPLE_ASTERISK_PATTERN.sub(
        _single_asterisk_replacement,
        protected_text,
    )
    protected_text = DOUBLE_ASTERISK_PATTERN.sub(
        _single_asterisk_replacement,
        protected_text,
    )
    protected_text = DOUBLE_UNDERSCORE_PATTERN.sub(
        _single_asterisk_replacement,
        protected_text,
    )

    protected_text = BROKEN_DOUBLE_OPEN_PATTERN.sub(
        _plain_text_replacement,
        protected_text,
    )
    protected_text = BROKEN_DOUBLE_CLOSE_PATTERN.sub(
        _plain_text_replacement,
        protected_text,
    )
    protected_text = BROKEN_UNDERSCORE_OPEN_PATTERN.sub(
        _plain_text_replacement,
        protected_text,
    )
    protected_text = BROKEN_UNDERSCORE_CLOSE_PATTERN.sub(
        _plain_text_replacement,
        protected_text,
    )

    protected_text, bold_tokens = _protect_valid_whatsapp_bold(protected_text)
    protected_text = protected_text.replace("**", "")
    protected_text = protected_text.replace("__", "")
    protected_text = re.sub(r"(?<!\d)\*(?!\d)", "", protected_text)

    protected_text = _restore_tokens(protected_text, bold_tokens, prefix="BOLDTOKEN")
    return _restore_tokens(protected_text, url_tokens, prefix="URLTOKEN")


def _normalize_label_colon_format(
    text: str,
    *,
    route: str | None = None,
    topic: str | None = None,
) -> str:
    lines = text.split("\n")
    allow_emoji = _is_loans_context(route=route, topic=topic) and _count_emojis(text) == 0
    added_emojis = 0
    normalized_lines: list[str] = []

    for line in lines:
        updated_line = line
        label_text: str | None = None
        match = LABEL_COLON_OUTSIDE_PATTERN.match(line) or LABEL_COLON_INSIDE_PATTERN.match(line)

        if match:
            raw_label = match.group("label").strip()
            if _looks_like_label(raw_label):
                label_text = raw_label.rstrip(":").strip()
                rest = match.group("rest").strip()
                emoji_prefix = match.group("emoji") or ""
                updated_line = f"{match.group('indent')}{emoji_prefix}*{label_text}:*"
                if rest:
                    updated_line = f"{updated_line} {rest}"

        if label_text and allow_emoji and added_emojis < 3 and match and not match.group("emoji"):
            emoji = _emoji_for_label(label_text)
            if emoji:
                indent = match.group("indent")
                content = updated_line[len(indent):]
                updated_line = f"{indent}{emoji} {content}"
                added_emojis += 1

        normalized_lines.append(updated_line)

    return "\n".join(normalized_lines)


def _add_discreet_emojis(
    text: str,
    *,
    topic: str | None = None,
    route: str | None = None,
) -> str:
    if not _is_loans_context(route=route, topic=topic):
        return text

    if _count_emojis(text) > 0:
        return text

    emoji = _select_primary_emoji(text)
    if not emoji:
        return text

    lines = text.split("\n")
    updated_lines = list(lines)

    for index, line in enumerate(lines):
        if not line.strip():
            continue

        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        updated_lines[index] = f"{indent}{emoji} {stripped}"
        break

    return "\n".join(updated_lines)


def _replace_markdown_link(match: re.Match[str]) -> str:
    label = match.group(1).strip()
    url = match.group(2).strip()
    return f"{label}: {url}" if label else url


def _single_asterisk_replacement(match: re.Match[str]) -> str:
    content = match.group(1).strip()
    return f"*{content}*" if content else ""


def _plain_text_replacement(match: re.Match[str]) -> str:
    return match.group(1).strip()


def _protect_urls(text: str) -> tuple[str, list[str]]:
    tokens: list[str] = []

    def replace(match: re.Match[str]) -> str:
        token = f"URLTOKEN{len(tokens)}PLACEHOLDER"
        tokens.append(match.group("url"))
        return token

    return URL_PATTERN.sub(replace, text), tokens


def _protect_valid_whatsapp_bold(text: str) -> tuple[str, list[str]]:
    tokens: list[str] = []

    def replace(match: re.Match[str]) -> str:
        token = f"BOLDTOKEN{len(tokens)}PLACEHOLDER"
        tokens.append(match.group(0))
        return token

    return VALID_WHATSAPP_BOLD_PATTERN.sub(replace, text), tokens


def _restore_tokens(text: str, tokens: list[str], *, prefix: str) -> str:
    restored = text
    for index, value in enumerate(tokens):
        restored = restored.replace(f"{prefix}{index}PLACEHOLDER", value)
    return restored


def _normalize_for_matching(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    without_accents = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    lowered = without_accents.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _looks_like_label(label: str) -> bool:
    cleaned = label.strip().strip(":").strip()
    if not cleaned:
        return False

    if len(cleaned) > 50:
        return False

    if any(char in cleaned for char in ".!?"):
        return False

    return len(cleaned.split()) <= 8


def _is_loans_context(
    *,
    route: str | None = None,
    topic: str | None = None,
) -> bool:
    normalized_route = _normalize_for_matching(route)
    normalized_topic = _normalize_for_matching(topic)

    if normalized_route in EMOJI_BLOCKED_ROUTES:
        return False

    if normalized_topic in EMOJI_BLOCKED_TOPICS:
        return False

    if normalized_route in LOANS_ROUTES:
        return True

    if normalized_topic in LOANS_TOPICS:
        return True

    return "prestamo" in normalized_topic


def _emoji_for_label(label: str) -> str | None:
    normalized = _normalize_for_matching(label)

    rules = (
        (("hipotecario", "vivienda", "casa", "hogar"), "🏠"),
        (("prendario", "auto", "vehiculo"), "🚗"),
        (("destino", "pasos", "solicitud", "como pedir", "como solicitar"), "📌"),
        (("tasas", "tasa", "cuotas", "cuota", "monto", "montos", "financiacion", "importe"), "💰"),
        (("documentacion", "requisitos", "requisito"), "📄"),
        (("app", "online banking", "canales", "canal"), "📱"),
        (("uva", "cotizacion", "informacion", "detalle"), "ℹ️"),
    )

    for keywords, emoji in rules:
        if any(keyword in normalized for keyword in keywords):
            return emoji

    return None


def _select_primary_emoji(text: str) -> str | None:
    normalized = _normalize_for_matching(text)

    if any(
        keyword in normalized
        for keyword in ("hipotecario", "vivienda", "casa", "hogar", "primera vivienda")
    ):
        return "🏠"

    if any(keyword in normalized for keyword in ("prendario", "auto", "vehiculo")):
        return "🚗"

    if normalized.startswith(
        ("para avanzar", "para pedir", "para solicitar", "si queres", "si queres avanzar")
    ):
        return "📌"

    if "informacion" in normalized or "detalle" in normalized:
        return "ℹ️"

    return "💰"


def _count_emojis(text: str) -> int:
    total = 0
    remaining = text

    for emoji in KNOWN_EMOJIS:
        occurrences = remaining.count(emoji)
        total += occurrences
        if occurrences:
            remaining = remaining.replace(emoji, "")

    total += len(EMOJI_REGEX.findall(remaining))
    return total
