# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

基于 LangChain + LangGraph 构建的校园智能问答与预订系统。支持 RAG 知识问答和 SQL Agent 场馆预订双能力，通过意图路由自动分发用户请求。

## Tech Stack

- **LLM**: Qwen 系列模型（阿里云 DashScope OpenAI 兼容接口）
- **框架**: FastAPI + LangChain + LangGraph
- **向量数据库**: Milvus (RAG 检索)
- **关系数据库**: MySQL (场馆/预订/消息)
- **缓存/状态**: Redis (LangGraph checkpoint + 记忆存储 via RedisSaver)
- **包管理**: uv (pyproject.toml)

## Key Commands

```bash
# 启动服务
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# 构建 Milvus 向量索引
uv run python rag/rag_loadDocuments.py

# 运行 LangSmith 评估
uv run python langsmith_eval.py
```

## Architecture

### LangGraph 主图 (graph.py)

采用**意图路由 + 子图**架构，所有节点共享 `MessagesState` 基类：

```
用户消息 → intent 节点 (意图分类) → sql_graph 或 rag_graph 子图
```

**意图分类器**：使用轻量模型 `qwen-turbo` 判断用户问题属于 `sql`（场馆预订业务）还是 `rag`（知识问答）。

**子图通用结构**（SQL/RAG 子图相同模式）：

```
agent ──(有 tool_calls)──▶ tools ──(达到 20 条消息)──▶ summarize ──▶ agent
  │                              │
  └──(无 tool_calls)─────────────▶ __end__
```

- **agent 节点**: 加载记忆摘要 + 未总结消息作为上下文，调用 `model_with_tools.invoke()`
- **tools 节点**: `ToolNode` 封装工具列表，工具结果自动回传给 agent
- **summarize 节点**: 每 10 条未总结消息触发一次，调用轻量模型生成摘要存入 RedisStore

### 记忆系统

- **存储**: RedisSaver (checkpoint) + RedisStore (摘要)
- **触发**: 每 10 条消息生成一次摘要（`summarize_and_store` 节点）
- **使用**: agent 节点从 RedisStore 读取 `("user", thread_id) -> "summary"` 作为对话上下文
- **目的**: 降低 token 消耗和模型幻觉

### 模块划分

| 模块 | 职责 | 关键文件 |
|------|------|----------|
| **RAG** | Milvus 向量检索 | `rag/rag_chain.py`, `rag/rag_loadDocuments.py`, `rag/rag_index_mivus.py` |
| **SQL Tools** | LangGraph 工具（场馆查询/预订 CRUD） | `sql/tools/tool1.py` |
| **SQL REST API** | FastAPI 场馆/预订接口 | `sql/api/venueApi.py`, `sql/api/bookingAPi.py` |
| **数据模型** | SQLAlchemy 表定义 + Pydantic 实体 | `sql/table/models.py`, `sql/entity/*.py` |
| **消息存储** | 对话记录异步入库 | `sql/crud/message.py` |

### API 入口 (main.py)

- `POST /chat` — 同步对话，`graph.invoke()` 返回最终 AI 消息
- `POST /stream/chat` — SSE 流式响应，`graph.stream()` 流式输出

### 注意事项

- **RedisSaver 是同步的**：只能用 `graph.stream()` 而非 `graph.astream()`，用 `invoke()` 而非 `ainvoke()`
- **stream_mode="messages"**：返回的 event 是 `{'type': 'messages', 'ns': (), 'data': (msg, metadata)}` 格式的 dict，需通过 `event["data"]` 解包
- **子图编译**：`sql_graph` 和 `rag_graph` 在 `graph.py` 模块加载时编译，共享 `redis_checkpointer` 和 `redis_store`
- **环境变量**: 通过 `.env` 配置，使用 `python-dotenv` 加载。必需变量：`DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`, `MYSQL_*`, `REDIS_HOST`, `REDIS_PORT`, `MILVUS_CONN_ALIAS`, `LANGSMITH_*`
