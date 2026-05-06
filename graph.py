import logging
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.redis import RedisStore
from langgraph.types import interrupt

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
    response = sql_model.invoke(context)
    return {"messages": [response]}


# ================== 人工检查节点 ==================
WRITE_TOOL_NAMES = {"cancel_booking", "create_booking"}


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

    return state  # 批准: 不做修改,继续走 ToolNode

# ==================SQL 子图=================
def sql_graph_builder():
    builder = StateGraph(AgentState)

    def _build_sql_graph_node(store: BaseStore):
        def sql_agent_node(state, config: RunnableConfig, *, store: BaseStore):
            msgs = state["messages"]
            last_idx = state.get("last_summarized_index", 0)
            summary_item = store.get(("user", config["configurable"]["thread_id"]), "summary")
            context = [SystemMessage(
                "你是福建师范大学场馆预订助手。你有以下工具: query_venues, check_availability, create_booking, cancel_booking。\n"
                "工作规则：\n"
                "1. 用户要求查询场地时,调用 query_venues 或 check_availability\n"
                "2. check_availability 接受场地名字(模糊匹配)和日期,返回结果中包含 venue_id\n"
                "3. 用户预约时,收集学生ID、场地名字、开始时间、结束时间。信息不全时向用户询问\n"
                "4. 收集齐后,先调用 check_availability 检查空闲,从返回结果中提取 venue_id,然后调用 create_booking(student_id, venue_id, start, end)\n"
                "5. 用户要求取消预约时,调用 cancel_booking\n"
                "6. 收到工具结果后,用简洁的自然语言回复用户,不要使用markdown表格或emoji"
            )]
            if summary_item:
                context.append(HumanMessage(summary_item.value['content']))
            context.extend(msgs[last_idx:])
            response = sql_model.invoke(context)
            return {"messages": [response]}

        return sql_agent_node

    def should_use_sql_tool(state):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    def route_after_agent(state: MessagesState) -> Literal["review", "tools", "__end__"]:
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage):
            logger.info(f"[SQL] route_after_agent: not AIMessage, ending")
            return "__end__"

        if not last_msg.tool_calls:
            logger.info(f"[SQL] route_after_agent: no tool_calls, ending")
            return "__end__"  # 没有工具调用，结束

        has_write = any(tc["name"] in WRITE_TOOL_NAMES for tc in last_msg.tool_calls)
        logger.info(f"[SQL] route_after_agent: tool_calls={[tc['name'] for tc in last_msg.tool_calls]}, has_write={has_write}")

        if has_write:
            logger.info(f"[SQL] routing to review")
            return "review"
        return "tools"

    builder.add_node("agent", _build_sql_graph_node(redis_store))
    builder.add_node("review",review_node)
    builder.add_node("tools", ToolNode(sql_tools))
    builder.add_node("summarize", summarize_and_store)

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools","review":"review", "__end__": "__end__"},
    )
    builder.add_conditional_edges(
        "tools",
        should_summarize,
        {True: "summarize", False: "agent"}
    )
    builder.add_edge("summarize", "agent")
    builder.add_edge("review","agent")
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
            context = [SystemMessage(
                "你是福建师范大学校园知识问答助手。使用 rag_search 工具检索校园知识库回答用户问题。\n"
                "回答要求：\n"
                "- 基于工具返回的检索结果回答，不要编造信息\n"
                "- 语言简洁，条理清晰，避免过度格式化（不用markdown表格、emoji等花哨排版）\n"
                "- 如果检索结果为空，坦诚告知用户知识库中没有相关信息\n"
            )]
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
        msgs = state["messages"]
        user_input = get_message_text(msgs[-1])

        # 获取最近的几条历史消息作为上下文，帮助分类
        recent = msgs[-5:]  # 最近5条
        history_text = format_messages(recent) if len(recent) > 1 else ""

        result = extractor_model.invoke([
            SystemMessage(
                "你是一个意图分类器。根据用户当前问题和上下文判断意图：\n"
                "- 如果上下文涉及场馆查询、预订、取消、可用性检查等业务操作，且当前消息是对之前业务对话的延续，返回: sql\n"
                "- 如果问题是闲聊、知识问答、通用问题，返回: rag\n"
                "只返回 sql 或 rag，不要返回其他内容。"
            ),
            HumanMessage(
                f"上下文对话历史:\n{history_text}\n\n"
                f"当前用户消息: {user_input}"
            ),
        ]).content.strip().lower()

        intent = result if result in ("sql", "rag") else "rag"
        logger.info(f"[intent] classified as: {intent}")
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
