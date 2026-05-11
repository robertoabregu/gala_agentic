import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_text(
    text: str,
    min_chunk_size: int = 500,
    max_chunk_size: int = 800,
) -> list[str]:
    cleaned_text = normalize_text(text)
    if not cleaned_text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
    if len(sentences) == 1 and len(cleaned_text) <= max_chunk_size:
        return [cleaned_text]

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.extend(_split_long_sentence(sentence, max_chunk_size))
            continue

        candidate = sentence if not current_chunk else f"{current_chunk} {sentence}"
        if len(candidate) <= max_chunk_size:
            current_chunk = candidate
            continue

        chunks.append(current_chunk)
        current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return _merge_small_tail(chunks, min_chunk_size, max_chunk_size)


def chunk_documents(
    documents: list[dict],
    min_chunk_size: int = 500,
    max_chunk_size: int = 800,
) -> list[dict]:
    chunked_documents: list[dict] = []

    for document in documents:
        text_chunks = split_text(
            document["content"],
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )

        for chunk_index, chunk_text in enumerate(text_chunks, start=1):
            chunked_documents.append({
                "chunk_id": f'{document["id"]}-chunk-{chunk_index}',
                "document_id": document["id"],

                "title": document.get("title", ""),
                "url": document.get("url", ""),
                "content": chunk_text,

                # NUEVA METADATA
                "product_type": document.get("product_type", "otro"),
                "document_purpose": document.get("document_purpose", "otro"),
                "topics": document.get("topics", []),

                "classification_confidence": document.get(
                    "classification_confidence",
                    0.0,
                ),

                "classification_reason": document.get(
                    "classification_reason",
                    "",
                ),
            })

    return chunked_documents


def _split_long_sentence(sentence: str, max_chunk_size: int) -> list[str]:
    return [
        sentence[index : index + max_chunk_size].strip()
        for index in range(0, len(sentence), max_chunk_size)
        if sentence[index : index + max_chunk_size].strip()
    ]


def _merge_small_tail(
    chunks: list[str],
    min_chunk_size: int,
    max_chunk_size: int,
) -> list[str]:
    if len(chunks) < 2:
        return chunks

    last_chunk = chunks[-1]
    previous_chunk = chunks[-2]

    if len(last_chunk) >= min_chunk_size:
        return chunks

    merged_chunk = f"{previous_chunk} {last_chunk}".strip()
    if len(merged_chunk) <= max_chunk_size:
        return [*chunks[:-2], merged_chunk]

    return chunks
