from pydantic import BaseModel, Field

from graph.rag.rag_config import query_rewrite_model
from graph.rag.state.rag_agent_state import RagAgentState


class RewrittenQuestions(BaseModel):
    questions: list[str] = Field(description="改写后的查询列表,2-3个变体")


_rewriter = query_rewrite_model.with_structured_output(RewrittenQuestions)


def query_rewrite(state: RagAgentState):
    question = state["question"]
    previous = state.get("rewritten_questions") or []

    previous_text = "\n".join(f"- {q}" for q in previous) if previous else "（无历史查询）"

    prompt = f"""你是电商搜索查询优化专家。用户的商品搜索效果不佳,需要改写成更适合商品库检索的形式。

            用户原始问题:
            {question}

            之前尝试过的查询:
            {previous_text}

            请生成2-3个新的查询变体,要求:
            - 从不同角度表达同一购买意图(同义词替换、具体化品牌/型号、拆解需求等)
            - 不要重复之前已经尝试过的查询
            - 适合用于商品库向量检索,包含明确的商品关键词"""

    result: RewrittenQuestions = _rewriter.invoke(prompt)

    return {
        "rewritten_questions": result.questions,
        "iterations": state.get("iterations", 0) + 1,
        "trace": [f"rewrite: 生成 {len(result.questions)} 个查询 -> {result.questions}"],
    }


