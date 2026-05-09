import logging
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore
from langgraph.store.redis import RedisStore
from langgraph.types import interrupt

from graph.rag.rag_previous_work.rag_chain import rag_search
from graph.sql.sql_previous_work.tools.tool1 import query_venues, check_availability, create_booking, cancel_booking

logger = logging.getLogger(__name__)

load_dotenv()

# ==================模型配置=================
model = init_chat_model(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model_provider="openai",
    model="qwen3.5-35b-a3b",
)
extractor_model = init_chat_model(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model_provider="openai",
    model="qwen-turbo",
)

# ==================Redis配置=================
REDIS_URL = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
redis_checkpointer = RedisSaver(
    redis_url=REDIS_URL,
    ttl={
        "default_ttl": 60,
        "refresh_on_read": False,
    },
)
redis_store = RedisStore(
    conn=redis_checkpointer._redis,
)
redis_store.setup()
redis_checkpointer.setup()

# ==================工具定义=================
tools = [
    rag_search, query_venues, check_availability,
    create_booking, cancel_booking
]
rag_tools = [rag_search]
sql_tools = [query_venues, check_availability, create_booking, cancel_booking]
sql_model = model.bind_tools(sql_tools)
rag_model = model.bind_tools(rag_tools)

# ==================状态定义=================
class AgentState(MessagesState):
    last_summarized_index: int = 0
    _intent: str = "rag"

# ==================常量=================
WRITE_TOOL_NAMES = {"cancel_booking", "create_booking"}

# ==================辅助函数=================
def get_message_text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in msg.content
        )
    return str(msg.content)


def format_messages(messages: list):
    parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "用户"
        elif isinstance(msg, AIMessage):
            role = "AI"
        elif isinstance(msg, ToolMessage):
            role = "工具结果"
        else:
            role = "未知"
        text = get_message_text(msg)
        parts.append(f"[{role}]: {text}")
    return "\n".join(parts)


def extract_relevant_memory(full_history: list, user_question: str) -> str:
    if not user_question:
        return "\n".join(full_history)
    prompt = f"""以下是完整对话历史,用户当前的问题是: "{user_question}"
请提取与当前问题相关的记忆,只保留有用的上下文,不相关的直接丢弃。
输出格式为简洁的摘要。
---
对话历史:
{format_messages(full_history)}
"""
    result = extractor_model.invoke([HumanMessage(prompt)])
    return result.content


# ==================摘要逻辑=================
def should_summarize(state):
    msgs = state.get("messages", [])
    last_idx = state.get("last_summarized_index", 0)
    return (len(msgs) - last_idx) >= 20


def get_last_summarized_index(store: BaseStore, thread_id: str, scope: str) -> int:
    """从 store 读取持久化的 last_summarized_index。"""
    item = store.get(("user", thread_id, scope), "last_index")
    if item:
        return item.value.get("index", 0)
    return 0


def make_summarize_and_store(scope: str):
    """为指定子图创建独立的 summarize 节点。"""
    def summarize_and_store(state, config: RunnableConfig, *, store: BaseStore):
        msgs = state.get("messages", [])
        thread_id = config["configurable"]["thread_id"]
        ns = ("user", thread_id, scope)

        # 倒序找上一个 HumanMessage，保留从它开始的原文
        keep_start = 0
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage):
                keep_start = i
                break

        to_summary = msgs[:keep_start]

        if to_summary:
            summary = extractor_model.invoke([
                SystemMessage("压缩以下对话为摘要，保留关键事实和决策。"),
                HumanMessage(format_messages(to_summary)),
            ]).content
            store.put(ns, "summary", {"content": summary})

        # 持久化 last_summarized_index
        store.put(ns, "last_index", {"index": keep_start})

        return {"last_summarized_index": keep_start}

    return summarize_and_store


def get_last_idx_from_store(state, config, scope: str):
    """统一从 store 读取 last_summarized_index，新服务启动时 state 里默认 0。"""
    state_idx = state.get("last_summarized_index", 0)
    thread_id = config["configurable"]["thread_id"]
    store_idx = get_last_summarized_index(redis_store, thread_id, scope)
    return max(state_idx, store_idx)


# ================== 人工检查节点 ==================
def review_node(state):
    """审查写操作,需要人工审批时暂停"""
    msg = state["messages"][-1]
    logger.info(f"[review] entered, msg type={type(msg).__name__}, tool_calls={msg.tool_calls if isinstance(msg, AIMessage) else 'N/A'}")
    if not isinstance(msg, AIMessage) or not msg.tool_calls:
        return state

    pending_writes = [tc for tc in msg.tool_calls if tc["name"] in WRITE_TOOL_NAMES]
    if not pending_writes:
        return state

    logger.info(f"[review] interrupting with pending_writes={pending_writes}")
    human_decision = interrupt({
        "pending_ops": [{"tool": tc["name"], "args": tc["args"]} for tc in pending_writes],
        "question": "确认执行以上数据库操作?请回复 yes 或 no",
    })

    answer = str(human_decision).strip().lower()
    if answer in ("no", "n", "reject", "false"):
        # 拒绝: 移除写操作,只保留查询工具
        read_only_calls = [tc for tc in msg.tool_calls if tc["name"] not in WRITE_TOOL_NAMES]
        return {"messages": [AIMessage(
            content=f"以下操作已被人工拒绝: {', '.join(tc['name'] for tc in pending_writes)}",
            tool_calls=read_only_calls,
            id=msg.id
        )]}

    return state
