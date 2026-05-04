# LangChain AI Learning

基于 LangChain + LangGraph 构建的校园智能问答与预订系统。支持 RAG 知识库问答和 SQL Agent 场馆预订双能力，通过意图路由自动分发用户请求。

## 技术栈

- **LLM**: Qwen3.5 (阿里云 DashScope)
- **框架**: FastAPI + LangChain + LangGraph
- **向量数据库**: Milvus (RAG 检索)
- **关系数据库**: MySQL (场馆/预订数据 + 消息记录)
- **缓存/状态**: Redis (LangGraph checkpoint + 记忆存储)
- **搜索**: Tavily API (网络搜索增强)
- **可观测性**: LangSmith (trace 追踪 + 评估)

## 项目结构

```
.
├── main.py                          # FastAPI 入口, /chat 和 /stream/chat 接口
├── graph.py                         # LangGraph 主图构建(意图路由 + 子图)
├── pyproject.toml                   # 项目依赖, uv 管理
├── .env                             # 环境变量(DashScope, MySQL, Redis, Milvus)
│
├── rag/                             # RAG 知识库模块
│   ├── rag_chain.py                 # rag_search 工具, Milvus 向量检索
│   ├── rag_loadDocuments.py         # 文档加载与 Milvus 索引构建
│   └── rag_index_mivus.py           # Milvus 集合管理
│
├── sql/                             # SQL Agent 模块
│   ├── api/                         # FastAPI REST 接口
│   │   ├── venueApi.py              # 场馆 CRUD
│   │   └── bookingAPi.py            # 预订 CRUD
│   ├── tools/
│   │   └── tool1.py                 # LangGraph 工具(query_venues, create_booking 等)
│   ├── entity/                      # Pydantic 请求/响应模型
│   ├── table/
│   │   └── models.py                # SQLAlchemy 表定义(venues, bookings, messages)
│   └── crud/
│       └── message.py               # 消息异步入库
│
├── college_rag_data/                # RAG 知识库原始文档
│   ├── 学校简介.md
│   ├── 图书馆使用指南.md
│   ├── 校园卡服务指南.md
│   ├── 教务处规章制度.md
│   └── 计算机与网络空间安全学院.md
│
├── learning-steps/                  # 学习文档归档
│   ├── LangGraphStreaming流式响应指南.md
│   ├── LangGraph人工确认(Human-in-the-Loop)学习指南.md
│   └── 学习文档/                    # Milvus, LangSmith, 记忆系统等专题指南
│
└── memory/                          # 本地持久化记忆(文件存储)
```

## 架构设计

### 意图路由主图

```
                    用户消息
                       │
                       ▼
              ┌─────────────────┐
              │  intent 节点     │  extractor_model 判断意图
              │  (意图分类)      │  sql 或 rag
              └────┬──────┬─────┘
                   │       │
              sql  │       │ rag
                   ▼       ▼
          ┌──────────┐ ┌──────────┐
          │sql_graph │ │ rag_graph│
          │  子图     │ │  子图    │
          └──────────┘ └──────────┘
```

### 子图结构 (SQL / RAG 通用)

```
agent ──(有 tool_calls)──▶ tools ──(达到 20 条消息)──▶ summarize ──▶ agent
  │                              │
  └──(无 tool_calls)─────────────▶ end
```

### 记忆系统

- 每 10 条未总结消息触发一次摘要
- 摘要存储到 RedisStore (`user/{thread_id}/summary`)
- 每次 agent 调用时加载摘要作为上下文
- 降低 token 消耗和模型幻觉

## 快速开始

### 1. 环境要求

- Python >= 3.12
- uv (包管理)
- MySQL >= 8.0
- Redis >= 6.0
- Milvus >= 2.4

### 2. 安装依赖

```bash
uv sync
```

### 3. 配置环境变量

复制 `.env` 并填写你的配置：

```env
# LLM (阿里云 DashScope)
DASHSCOPE_API_KEY=your_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# LangSmith 可观测性
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=langchain-ai-learning

# MySQL
MYSQL_HOST=192.168.56.10
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=root
MYSQL_DATABASE=lfl-college

# Redis
REDIS_HOST=192.168.56.10
REDIS_PORT=6379

# Milvus
MILVUS_CONN_ALIAS=default
```

### 4. 启动服务

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. 访问 API 文档

打开浏览器访问 http://127.0.0.1:8000/docs

## API 接口

### 普通对话

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"id": "user-1", "content": "福建师范大学计算机学院有哪些专业？"}'
```

### 流式对话

```bash
curl -X POST http://127.0.0.1:8000/stream/chat \
  -H "Content-Type: application/json" \
  -d '{"id": "user-1", "content": "帮我查一下明天体育馆可用吗？"}'
```

### 场馆 CRUD

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/venues` | 创建场馆 |
| GET | `/venues` | 查询所有场馆 |
| GET | `/venues/{id}` | 查询单个场馆 |
| PUT | `/venues/{id}` | 更新场馆 |
| DELETE | `/venues/{id}` | 删除场馆 |

### 预订 CRUD

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/bookings` | 创建预订 |
| GET | `/bookings` | 查询所有预订 |
| GET | `/bookings/{id}` | 查询单个预订 |
| PUT | `/bookings/{id}` | 更新预订 |
| DELETE | `/bookings/{id}` | 删除预订 |

## RAG 知识库

### 构建索引

将 markdown 文档放入 `college_rag_data/` 目录，然后运行：

```bash
uv run python rag/rag_loadDocuments.py
```

### 支持的文档格式

- Markdown (.md)
- 每篇文档自动分块(chunk)并向量化存储到 Milvus

## 评估

使用 LangSmith 进行 RAG 质量评估：

```bash
uv run python langsmith_eval.py
```

评估指标：
- **answer_relevance**: 回答是否解决了用户问题
- **faithfulness**: 回答是否基于检索上下文(无幻觉)
- **context_relevance**: 检索内容是否紧扣问题

