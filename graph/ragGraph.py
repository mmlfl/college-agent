from typing import Literal

from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from graph.graphConfig import (
    AgentState, redis_store, rag_model, rag_tools,
    get_last_idx_from_store, make_summarize_and_store,
    logger,
)


def rag_graph_builder():
    builder = StateGraph(AgentState)

    def _build_rag_graph_node(store):
        def rag_agent_node(state, config: RunnableConfig, *, store):
            msgs = state["messages"]
            last_idx = get_last_idx_from_store(state, config, "rag")
            summary_item = store.get(
                ("user", config["configurable"]["thread_id"], "rag"), "summary"
            )
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
            response = rag_model.invoke(context)
            return {"messages": [response]}

        return rag_agent_node

    def route_after_rag_agent(state: MessagesState) -> Literal["tools", "summarize", "__end__"]:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"

        # 无工具调用 = 对话自然结束，检查是否需要摘要
        msgs = state.get("messages", [])
        last_idx = state.get("last_summarized_index", 0)
        if len(msgs) - last_idx >= 20:
            logger.info(f"[RAG] conversation ended, {len(msgs) - last_idx} unsummarized msgs -> summarize")
            return "summarize"
        return "__end__"

    builder.add_node("agent", _build_rag_graph_node(redis_store))
    builder.add_node("tools", ToolNode(rag_tools))
    builder.add_node("summarize", make_summarize_and_store("rag"))

    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        route_after_rag_agent,
        {"tools": "tools", "summarize": "summarize", "__end__": "__end__"}
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("summarize", "__end__")

    return builder


