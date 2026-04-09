from __future__ import annotations

from src.rag.schema import DocumentChunk


def split_text_into_chunks(
    source: str,
    text: str,
    chunk_size: int = 350,
    chunk_overlap: int = 80,
) -> list[DocumentChunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    chunks: list[DocumentChunk] = []
    start = 0
    idx = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk_text = normalized[start:end].strip()
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source}::chunk-{idx}",
                    source=source,
                    content=chunk_text,
                )
            )
            idx += 1

        if end >= len(normalized):
            break
        start = end - chunk_overlap

    return chunks
