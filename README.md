# RAG + ReAct 扫地机器人 Agent

基于本地知识库（`data/`）构建的智能问答项目，支持：

- RAG 检索增强：从文档中检索证据再回答
- ReAct Agent：按 Thought / Action / Observation 进行工具调用
- 多入口运行：CLI、FastAPI、Streamlit 前端

## 1. 项目结构

```text
.
├── configs/
│   └── settings.yaml
├── data/
├── frontend/
│   ├── streamlit_app.py
│   └── README.md
├── scripts/
│   ├── build_index.py
│   └── chat_cli.py
├── src/
│   ├── agents/
│   ├── api/
│   ├── core/
│   ├── prompts/
│   ├── rag/
│   └── tools/
├── storage/
│   ├── index/
│   └── logs/
├── tests/
├── .env.example
└── requirements.txt
```

## 2. 环境准备

### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

### 2.2 配置环境变量

复制 `.env.example` 为 `.env`，并填写：

```bash
DASHSCOPE_API_KEY=your_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL_NAME=qwen3-max
EMBEDDING_MODEL_NAME=text-embedding-v4
```

说明：

- 你当前项目已适配 `qwen3-max`（对话模型）
- `EMBEDDING_MODEL_NAME` 已配置项就位（当前检索默认仍使用本地 TF-IDF）
- 若未配置 API Key，会自动回退到 Mock LLM（便于本地联调）

## 3. 运行步骤

### 3.1 构建知识库索引（首次必做）

```bash
python scripts/build_index.py
```

### 3.2 启动命令行对话

```bash
python scripts/chat_cli.py
```

### 3.3 启动 FastAPI 服务

```bash
uvicorn src.api.main:app --reload --port 8000
```

接口：

- `GET /health`
- `POST /chat`

请求示例：

```json
{
  "query": "拖布有异味怎么办？",
  "session_id": "user-001"
}
```

### 3.4 启动 Streamlit 前端

```bash
streamlit run frontend/streamlit_app.py
```

## 4. Streamlit 页面说明

- 聊天问答区：与 Agent 多轮对话
- 推理轨迹区：展开查看每轮 ReAct 轨迹
- 侧边栏控制：
  - 索引状态查看
  - 一键重建知识库索引
  - 清空会话历史

## 5. 已验证状态（2026-04-07）

在当前环境已完成以下验证：

1. `python scripts/build_index.py` 执行成功（索引 chunk 数 = 221）
2. `python -m pytest tests -q -p no:cacheprovider` 通过（2 passed）
3. CLI 调用 Agent 成功返回答案
4. Streamlit 可启动（`http://localhost:8501`）

## 6. 常见问题

1. 提示找不到索引文件：先执行 `python scripts/build_index.py`
2. 未配置 API Key：系统会自动使用 Mock LLM
3. 端口占用：

```bash
streamlit run frontend/streamlit_app.py --server.port 8502
```

## 7. 后续升级建议

1. 将 TF-IDF 检索替换为 `text-embedding-v4` + 向量数据库（FAISS/Milvus）
2. 增加设备状态工具（电量、错误码、地图状态）
3. 增加评测集与自动化回归（检索命中率、答案质量）
