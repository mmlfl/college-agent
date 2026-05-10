from langchain_core.messages import SystemMessage, HumanMessage

from graph.main.graph_config import rag_model
from graph.rag.state.rag_agent_state import RagAgentState

def generate_answer(state:RagAgentState):
    contexts = state.get("context") or []
    context_parts = []
    for i, c in enumerate(contexts, 1):
        tag = "商品描述" if c.get("doc_type") == "description" else "用户评价"
        context_parts.append(
            f"[{i}] 【{c.get('product_name', '未知商品')}】{tag}:\n{c['text']}"
        )
    context_text = "\n\n".join(context_parts) if context_parts else "（未检索到相关内容）"

    prompt = f"""你是电商智能导购助手。

     检索到的商品信息:
     {context_text}

     用户问题:
     {state['question']}

     回答要求：
     - 仅基于上下文回答，不要编造价格、参数等信息
     - 如果上下文为空或不足以回答，坦诚告知用户
     - 回答要突出商品核心卖点，帮助用户做购买决策
     - 语言简洁，条理清晰"""

    response = rag_model.invoke([HumanMessage(prompt)])
    return {
        "answer": response.content,
        "trace": [f"generate_answer: 生成长度 {len(response.content)} 的回答"],
    }


