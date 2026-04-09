from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.react_agent import ReActAgent
from src.core.config import load_settings
from src.core.llm import build_llm
from src.rag.pipeline import RAGConfig, RAGPipeline
from src.tools.knowledge_tool import KnowledgeTool
from src.tools.registry import ToolRegistry
from src.tools.robot_tool import RobotActionTool


def _inject_style() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap');
            :root {
                --app-text: #12263a;
                --app-muted: #28445d;
                --panel-bg: rgba(255, 255, 255, 0.78);
            }
            .stApp,
            [data-testid="stAppViewContainer"] {
                background: radial-gradient(1200px 700px at 8% -10%, #c8f0ff 0%, transparent 50%),
                            radial-gradient(900px 550px at 100% -20%, #ffe3c2 0%, transparent 50%),
                            linear-gradient(180deg, #f6f8fb 0%, #eef3f7 100%);
            }
            html, body, [class*="css"] {
                font-family: "Space Grotesk", "Noto Sans SC", sans-serif;
                color: var(--app-text);
            }
            p, li, label, span, div, h1, h2, h3, h4 {
                color: var(--app-text);
            }
            [data-testid="stSidebar"] {
                background: rgba(246, 249, 252, 0.92);
            }
            [data-testid="stSidebar"] *,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] div {
                color: #16314a !important;
            }
            [data-testid="stSidebar"] [data-baseweb="button"],
            [data-testid="stSidebar"] button {
                background: #eaf2fb !important;
                color: #16314a !important;
                border: 1px solid #bfd4ea !important;
            }
            [data-testid="stSidebar"] [data-baseweb="button"]:hover,
            [data-testid="stSidebar"] button:hover {
                background: #dfeaf7 !important;
                border-color: #a9c4df !important;
            }
            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] textarea,
            [data-testid="stSidebar"] [data-baseweb="input"] > div {
                background: #ffffff !important;
                color: #16314a !important;
                border-color: #bfd4ea !important;
            }
            [data-testid="stSidebar"] [data-testid="stAlert"] {
                background: #f4f8fd !important;
                border: 1px solid #bfd4ea !important;
            }
            [data-testid="stSidebar"] pre,
            [data-testid="stSidebar"] code {
                background: #eef4fb !important;
                color: #16314a !important;
                border-radius: 8px;
            }
            .hero {
                padding: 1rem 1.2rem;
                border-radius: 14px;
                background: var(--panel-bg);
                border: 1px solid rgba(18, 38, 58, 0.12);
                backdrop-filter: blur(8px);
                margin-bottom: 0.8rem;
            }
            .hero h1 {
                margin: 0;
                font-size: 1.6rem;
                letter-spacing: 0.2px;
            }
            .hero p {
                margin: 0.5rem 0 0 0;
                color: var(--app-muted);
                font-size: 0.95rem;
            }
            .chip {
                display: inline-block;
                margin-right: 0.5rem;
                margin-top: 0.4rem;
                background: #e5eef7;
                color: #12375a;
                border: 1px solid #c7d7e8;
                border-radius: 999px;
                padding: 0.18rem 0.7rem;
                font-size: 0.78rem;
                font-weight: 600;
            }
            [data-testid="stChatMessage"] {
                background: var(--panel-bg);
                border: 1px solid rgba(18, 38, 58, 0.12);
                border-radius: 14px;
                padding: 0.55rem 0.8rem;
                margin-bottom: 0.5rem;
            }
            [data-testid="stChatMessageContent"] *,
            [data-testid="stMarkdownContainer"] * {
                color: var(--app-text) !important;
            }
            [data-testid="stCodeBlock"] pre,
            pre, code {
                color: #0f2740 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _index_exists(index_dir: Path) -> bool:
    return (
        (index_dir / "chunks.json").exists()
        and (index_dir / "matrix.pkl").exists()
        and (index_dir / "vectorizer.pkl").exists()
    )


def _build_index() -> tuple[bool, str]:
    settings = load_settings()
    rag_cfg = RAGConfig(**settings.rag)
    rag = RAGPipeline(config=rag_cfg, index_dir=settings.index_dir)
    count = rag.build_index(settings.data_dir)
    return True, f"索引构建完成，chunk 数量: {count}"


@st.cache_resource(show_spinner=False)
def _build_agent() -> ReActAgent:
    settings = load_settings()
    rag_cfg = RAGConfig(**settings.rag)
    rag = RAGPipeline(config=rag_cfg, index_dir=settings.index_dir)
    rag.load_index()

    knowledge_tool = KnowledgeTool(rag=rag)
    action_tool = RobotActionTool()

    registry = ToolRegistry()
    registry.register(knowledge_tool.name, knowledge_tool.description, knowledge_tool.run)
    registry.register(action_tool.name, action_tool.description, action_tool.run)

    llm = build_llm(allow_mock_fallback=True)

    return ReActAgent(
        llm=llm,
        tool_registry=registry,
        system_prompt_path=Path("src/prompts/system_prompt.txt"),
        max_steps=settings.agent["max_steps"],
        temperature=settings.agent["temperature"],
    )


def _render_header() -> None:
    settings = load_settings()
    model_name = settings.models.get("chat_model_name", "qwen3-max")
    emb_name = settings.models.get("embedding_model_name", "text-embedding-v4")

    st.markdown(
        f"""
        <div class="hero">
            <h1>RAG + ReAct 扫地机器人 Agent</h1>
            <p>面向故障排查、维护保养、选购咨询的智能问答页面。</p>
            <span class="chip">Chat: {model_name}</span>
            <span class="chip">Embedding: {emb_name}</span>
            <span class="chip">Framework: Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(agent: ReActAgent) -> None:
    settings = load_settings()
    st.sidebar.title("控制台")
    st.sidebar.caption("这里用于索引状态与会话操作。")

    index_ok = _index_exists(settings.index_dir)
    st.sidebar.write(f"索引目录: `{settings.index_dir}`")
    st.sidebar.write(f"数据目录: `{settings.data_dir}`")
    st.sidebar.write("索引状态: 已就绪" if index_ok else "索引状态: 未检测到")

    st.sidebar.divider()
    st.sidebar.write(f"LLM 后端: `{getattr(agent.llm, 'backend_name', 'unknown')}`")
    st.sidebar.write(f"LLM 模型: `{getattr(agent.llm, 'model', 'unknown')}`")

    if st.sidebar.button("重建知识库索引", use_container_width=True):
        with st.spinner("正在构建索引..."):
            ok, msg = _build_index()
            _build_agent.clear()
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)

    if st.sidebar.button("重置 Agent 缓存", use_container_width=True):
        _build_agent.clear()
        st.sidebar.success("已清除 Agent 缓存，将在下次提问时重新初始化。")

    if st.sidebar.button("清空会话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def _ensure_index_or_warn() -> bool:
    settings = load_settings()
    if _index_exists(settings.index_dir):
        return True

    st.warning("未检测到本地索引，请先点击侧边栏“重建知识库索引”。")
    return False


def main() -> None:
    st.set_page_config(
        page_title="Robot Agent Frontend",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_style()
    _render_header()

    try:
        agent = _build_agent()
    except Exception as e:
        st.error(f"Agent 初始化失败: {e}")
        st.stop()

    _render_sidebar(agent)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("traces"):
                with st.expander("查看 ReAct 推理轨迹"):
                    for t in message["traces"]:
                        st.code(t)

    user_input = st.chat_input("请输入问题，例如：拖布有异味怎么处理？")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not _ensure_index_or_warn():
        return

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在检索与推理..."):
            stream_result = agent.run_stream(user_input)

        rendered_text = st.write_stream(stream_result.stream)
        with st.expander("查看 ReAct 推理轨迹"):
            for t in stream_result.traces:
                st.code(t)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": rendered_text,
            "traces": stream_result.traces,
        }
    )


if __name__ == "__main__":
    main()
