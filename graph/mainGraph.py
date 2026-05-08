import logging

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from graph.graphConfig import (
    AgentState, redis_checkpointer, redis_store,
    extractor_model, format_messages, get_message_text,
    should_summarize, make_summarize_and_store,
)
from graph.sql.sqlGraph import sql_graph_builder
from graph.ragGraph import rag_graph_builder

logger = logging.getLogger(__name__)


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
    from langchain_core.messages import AIMessage

    builder = StateGraph(AgentState)
    builder.add_node("summarize", make_summarize_and_store("default"))
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

rag_graph = rag_graph_builder().compile(redis_checkpointer,store=redis_store)
graph = main_graph_builder().compile(redis_checkpointer, store=redis_store)
