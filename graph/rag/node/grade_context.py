from pydantic import BaseModel, Field

from graph.rag.rag_config import grade_model
from graph.rag.state.rag_agent_state import RagAgentState


class GradeContextResult(BaseModel):
    is_relevant: bool = Field(description="是否与问题相关")
    score: float = Field(ge=0.0, le=1.0, description="相关性分数 0-1")
    reason: str = Field(description="评分理由")


_grader = grade_model.with_structured_output(GradeContextResult)


def grade_context_from_retrieve(state: RagAgentState):
    question = state["question"]
    contexts = state.get("context", [])
    context_text = "\n---\n".join(c["text"] for c in contexts)

    prompt = f"""你是检查官,评估检索到的商品信息/评价与用户购买咨询的相关性。

用户问题:
{question}

检索到的商品信息:
{context_text}

请评估商品信息/评价与用户购买意图的相关性。"""

    result: GradeContextResult = _grader.invoke(prompt)

    return {
        "context_grade": result,
        "trace": [f"grade_context: score={result.score}, relevant={result.is_relevant}"],
    }
