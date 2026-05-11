def build_context(results: list[dict]) -> str:
    context_blocks = []

    for index, result in enumerate(results, start=1):
        title = result.get("title") or "Sin título"
        source = result.get("url") or "Sin URL"
        content = result.get("content", "")

        context_blocks.append(
            f"[Chunk {index}]\n"
            f"Título: {title}\n"
            f"Fuente: {source}\n"
            f"Contenido: {content}"
        )

    return "\n\n".join(context_blocks)