# Streamlit 前端运行说明

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 构建知识库索引（首次必做）

```bash
python scripts/build_index.py
```

## 3. 启动前端

```bash
streamlit run frontend/streamlit_app.py
```

## 4. 页面功能

- 聊天问答：基于 RAG + ReAct 回答扫地机器人问题
- 会话历史：保留当前页面内的历史对话
- 推理轨迹：每轮可展开查看 Thought/Action/Observation
- 索引控制：侧边栏一键重建索引、清空会话
