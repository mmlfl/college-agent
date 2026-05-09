from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from graph.main.graph_config import (
    AgentState, redis_store, sql_model, sql_tools,
    WRITE_TOOL_NAMES, review_node,
    get_last_idx_from_store, make_summarize_and_store,
    logger,
)


def sql_graph_builder():
    builder = StateGraph(AgentState)

    def _build_sql_graph_node(store):
        def sql_agent_node(state, config: RunnableConfig, *, store):
            msgs = state["messages"]
            last_idx = get_last_idx_from_store(state, config, "sql")
            summary_item = store.get(
                ("user", config["configurable"]["thread_id"], "sql"), "summary"
            )
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

    def route_after_agent(state: MessagesState) -> Literal["review", "tools", "summarize", "__end__"]:
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage):
            logger.info(f"[SQL] route_after_agent: not AIMessage, ending")
            return "__end__"

        if last_msg.tool_calls:
            has_write = any(tc["name"] in WRITE_TOOL_NAMES for tc in last_msg.tool_calls)
            logger.info(f"[SQL] route_after_agent: tool_calls={[tc['name'] for tc in last_msg.tool_calls]}, has_write={has_write}")
            if has_write:
                logger.info(f"[SQL] routing to review")
                return "review"
            return "tools"

        # 无工具调用 = 对话自然结束，检查是否需要摘要
        msgs = state.get("messages", [])
        last_idx = state.get("last_summarized_index", 0)
        if len(msgs) - last_idx >= 20:
            logger.info(f"[SQL] conversation ended, {len(msgs) - last_idx} unsummarized msgs -> summarize")
            return "summarize"
        logger.info(f"[SQL] conversation ended, {len(msgs) - last_idx} unsummarized msgs (below threshold)")
        return "__end__"

    def route_after_review(state: MessagesState) -> Literal["tools", "__end__"]:
        """review 拒绝后结束，批准后执行工具"""
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"
        return "__end__"

    builder.add_node("agent", _build_sql_graph_node(redis_store))
    builder.add_node("review", review_node)
    builder.add_node("tools", ToolNode(sql_tools))
    builder.add_node("summarize", make_summarize_and_store("sql"))

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "review": "review", "summarize": "summarize", "__end__": "__end__"},
    )
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {"tools": "tools", "__end__": "__end__"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("summarize", "__end__")
    builder.set_entry_point("agent")
    return builder
