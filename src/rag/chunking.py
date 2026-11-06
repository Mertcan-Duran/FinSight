def chunk_text(
    text: str,
    chunk_size: int = 3000,
    overlap: int = 400,
):
    if chunk_size <= overlap:
        raise ValueError(
            "chunk_size must be greater than overlap"
        )

    chunks = []

    start = 0
    chunk_id = 0
    text_length = len(text)

    while start < text_length:
        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end]

        # Avoid cutting in the middle of a line
        if end < text_length:
            last_newline = chunk.rfind("\n")

            if last_newline > chunk_size // 2:
                end = start + last_newline
                chunk = text[start:end]

        cleaned_chunk = chunk.strip()

        if cleaned_chunk:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": cleaned_chunk,
                }
            )

            chunk_id += 1

        # Critical: stop after the final chunk
        if end >= text_length:
            break

        start = end - overlap

    return chunks