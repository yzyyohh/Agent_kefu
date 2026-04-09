from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    content: str

    def to_dict(self) -> dict:
        return asdict(self)
