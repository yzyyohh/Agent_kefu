from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_settings
from src.core.logger import get_logger
from src.rag.pipeline import RAGConfig, RAGPipeline


def main() -> None:
    settings = load_settings()
    logger = get_logger("build_index", log_dir=settings.log_dir)

    rag_cfg = RAGConfig(**settings.rag)
    rag = RAGPipeline(config=rag_cfg, index_dir=settings.index_dir)
    count = rag.build_index(settings.data_dir)

    logger.info("Index built successfully, chunks=%s, index_dir=%s", count, settings.index_dir)


if __name__ == "__main__":
    main()
