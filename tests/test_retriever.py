from __future__ import annotations

from pathlib import Path

from src.rag.pipeline import RAGConfig, RAGPipeline


def test_retriever_basic() -> None:
    cfg = RAGConfig(chunk_size=80, chunk_overlap=20, top_k=2, min_score=0.01)
    index_dir = Path("storage/index/test_tmp")
    data_dir = Path("tests/tmp_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "doc.txt").write_text("扫地机器人拖布异味处理：清洗拖布并晾干。", encoding="utf-8")

    pipeline = RAGPipeline(config=cfg, index_dir=index_dir)
    pipeline.build_index(data_dir)
    hits = pipeline.retrieve("拖布异味")

    assert len(hits) >= 1
    assert "拖布" in hits[0].chunk.content
