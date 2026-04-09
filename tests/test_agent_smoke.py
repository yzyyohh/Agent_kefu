from __future__ import annotations

from pathlib import Path

from src.agents.react_agent import ReActAgent
from src.core.llm import MockLLM
from src.rag.pipeline import RAGConfig, RAGPipeline
from src.tools.knowledge_tool import KnowledgeTool
from src.tools.registry import ToolRegistry
from src.tools.robot_tool import RobotActionTool


def test_agent_smoke() -> None:
    data_dir = Path("tests/tmp_data_agent")
    index_dir = Path("storage/index/test_agent")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "faq.txt").write_text("维护建议：定期清理滤网和滚刷。", encoding="utf-8")

    rag = RAGPipeline(config=RAGConfig(top_k=2, min_score=0.01), index_dir=index_dir)
    rag.build_index(data_dir)

    registry = ToolRegistry()
    knowledge = KnowledgeTool(rag=rag)
    robot_tool = RobotActionTool()

    registry.register(knowledge.name, knowledge.description, knowledge.run)
    registry.register(robot_tool.name, robot_tool.description, robot_tool.run)

    agent = ReActAgent(
        llm=MockLLM(),
        tool_registry=registry,
        system_prompt_path=Path("src/prompts/system_prompt.txt"),
        max_steps=3,
        temperature=0.2,
    )

    result = agent.run("拖布有异味怎么办")
    assert result.answer
