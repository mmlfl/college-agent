import logging
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.redis import RedisStore

from rag.rag_chain import rag_search
from sql.tools.tool1 import query_venues, check_availability, create_booking, cancel_booking

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
        "default_ttl": 10080,
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
model_with_tools = model.bind_tools(tools)


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


# ==================状态定义=================
class AgentState(MessagesState):
    last_summarized_index: int = 0
    _intent: str = "rag"


# ==================摘要逻辑=================
def should_summarize(state):
    msgs = state.get("messages", [])
    last_idx = state.get("last_summarized_index", 0)
    return (len(msgs) - last_idx) >= 20


def summarize_and_store(state, config: RunnableConfig, *, store: BaseStore):
    msgs = state.get("messages", [])
    last_index = state.get("last_summarized_index", 0)
    to_summary = msgs[last_index:last_index + 10]
    summary = extractor_model.invoke([
        SystemMessage("压缩以下对话为摘要:"),
        HumanMessage(format_messages(to_summary)),
    ]).content
    store.put(
        ("user", config["configurable"]["thread_id"]),
        "summary",
        {"content": summary},
    )
    return {"last_summarized_index": last_index + len(to_summary)}


# ==================通用 agent 节点=================
def agent_node(state, config: RunnableConfig, *, store: BaseStore):
    msgs = state["messages"]
    last_idx = state.get("last_summarized_index", 0)
    summary_item = store.get(("user", config["configurable"]["thread_id"]), "summary")
    summary = summary_item.value["content"] if summary_item else None
    unsummarized = msgs[last_idx:]
    if summary:
        context = [HumanMessage(summary)] + unsummarized
    else:
        context = unsummarized
    response = model_with_tools.invoke(context)
    return {"messages": [response]}


# ==================SQL 子图=================
def sql_graph_builder():
    builder = StateGraph(AgentState)

    def _build_sql_graph_node(store: BaseStore):
        def sql_agent_node(state, config: RunnableConfig, *, store: BaseStore):
            msgs = state["messages"]
            last_idx = state.get("last_summarized_index", 0)
            summary_item = store.get(("user", config["configurable"]["thread_id"]), "summary")
            context = []
            if summary_item:
                context.append(HumanMessage(f"记忆摘要:\n{summary_item.value['content']}"))
            context.extend(msgs[last_idx:])
            response = model.invoke(context)
            return {"messages": [response]}

        return sql_agent_node

    def should_use_sql_tool(state):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    builder.add_node("agent", _build_sql_graph_node(redis_store))
    builder.add_node("tools", ToolNode(sql_tools))
    builder.add_node("summarize", summarize_and_store)

    builder.add_conditional_edges(
        "agent",
        should_use_sql_tool,
        {"tools": "tools", "__end__": "__end__"},
    )
    builder.add_conditional_edges(
        "tools",
        should_summarize,
        {True: "summarize", False: "agent"}
    )
    builder.add_edge("summarize", "agent")
    builder.set_entry_point("agent")
    return builder


# ==================RAG 子图=================
def rag_graph_builder():
    builder = StateGraph(AgentState)

    def _build_rag_graph_node(store: BaseStore):
        def rag_agent_node(state, config: RunnableConfig, *, store: BaseStore):
            msgs = state["messages"]
            last_idx = state.get("last_summarized_index", 0)
            summary_item = store.get(("user", config["configurable"]["thread_id"]), "summary")
            context = []
            if summary_item:
                context.append(SystemMessage(f"记忆摘要:\n{summary_item.value['content']}"))
            context.extend(msgs[last_idx:])
            response = model.invoke(context)
            return {"messages": [response]}

        return rag_agent_node

    def should_use_rag_tool(state):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"
    builder.add_node("agent", _build_rag_graph_node(redis_store))
    builder.add_node("tools", ToolNode(rag_tools))
    builder.add_node("summarize", summarize_and_store)

    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_use_rag_tool,
        {"tools": "tools", "__end__": "__end__"}
    )
    builder.add_conditional_edges(
        "tools",
        should_summarize,
        {True: "summarize", False: "agent"}
    )
    builder.add_edge("summarize", "agent")

    return builder


# ==================意图路由主图=================
def main_graph_builder():
    builder = StateGraph(AgentState)
    sql_g = sql_graph_builder().compile(redis_checkpointer, store=redis_store)
    rag_g = rag_graph_builder().compile(redis_checkpointer, store=redis_store)

    def intent_router(state, config: RunnableConfig):
        msg = state["messages"][-1]
        user_input = get_message_text(msg)

        result = extractor_model.invoke([
            SystemMessage(
                "你是一个意图分类器。判断用户问题的意图：\n"
                "- 如果问题涉及场馆查询、预订、取消、可用性检查、场地管理等业务操作，返回: sql\n"
                "- 如果问题是闲聊、知识问答、通用问题，返回: rag\n"
                "只返回 sql 或 rag，不要返回其他内容。"
            ),
            HumanMessage(user_input),
        ]).content.strip().lower()

        intent = result if result in ("sql", "rag") else "rag"
        return {"_intent": intent}

    builder.add_node("sql_graph", sql_g)
    builder.add_node("rag_graph", rag_g)
    builder.add_node("intent", intent_router)

    builder.set_entry_point("intent")
    builder.add_conditional_edges(
        "intent",
        lambda s: s.get("_intent", "rag"),
        {"sql": "sql_graph", "rag": "rag_graph"},
    )
    return builder


# ==================扁平图(当前使用)=================
def graph_build():
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("summarize", summarize_and_store)
    builder.set_entry_point("agent")

    def should_use_tool(state) -> str:
        last = state['messages'][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    builder.add_conditional_edges(
        "agent",
        should_use_tool,
        {"tools": "tools", "__end__": "__end__"},
    )
    builder.add_conditional_edges(
        "tools",
        should_summarize,
        {True: "summarize", False: "agent"}
    )
    builder.add_edge("summarize", "agent")
    return builder.compile(checkpointer=redis_checkpointer, store=redis_store)


graph = main_graph_builder().compile(redis_checkpointer, store=redis_store)
