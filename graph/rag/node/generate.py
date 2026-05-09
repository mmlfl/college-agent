from langchain_core.messages import SystemMessage, HumanMessage

from graph.main.graph_config import rag_model
from graph.rag.state.rag_agent_state import RagAgentState

def generate_answer(state:RagAgentState):
    contexts = state.get("context") or []
    context_parts = [
        f"[来源: {c['source']}]\n{c['text']}"
        for c in contexts
    ]
    context_text = "\n\n".join(context_parts) if context_parts else "（未检索到相关内容）"
    prompt = f"""你是福建师范大学校园知识问答助手。

     检索到的上下文:
     {context_text}

     用户问题:
     {state['question']}

     回答要求：
     - 仅基于上下文回答，不要编造信息
     - 如果上下文为空或不足以回答问题，坦诚告知用户
     - 语言简洁，条理清晰"""

    response = rag_model.invoke([HumanMessage(prompt)])
    return {
        "answer": response.content,
        "trace": [f"generate_answer: 生成长度 {len(response.content)} 的回答"],
    }


