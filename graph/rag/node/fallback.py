from graph.rag.state.rag_agent_state import RagAgentState


def fallback(state: RagAgentState):
    return {
        "answer": "抱歉,经过多次检索和分析,当前知识库中未能找到足够的信息来回答您的问题。",
        "trace": [f"fallback: 已达到最大迭代次数({MAX_ITERATIONS}),终止检索"],
    }
