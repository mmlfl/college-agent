from graph.rag.state.rag_agent_state import RagAgentState


def fallback(state: RagAgentState):
    return {
        "answer": "抱歉,经过多次检索,商品库中未能找到匹配的信息。请尝试换个关键词或缩小搜索范围。",
        "trace": [f"fallback: 已达到最大迭代次数({MAX_ITERATIONS}),终止检索"],
    }
