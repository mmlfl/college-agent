from pydantic import BaseModel, Field

from graph.rag.rag_config import grade_model
from graph.rag.state.rag_agent_state import RagAgentState


class GradeAnswerResult(BaseModel):
    is_faithful: bool = Field(description="答案是否忠诚于上下文")
    score: float = Field(ge=0.0, le=1.0, description="忠诚度 0-1")
    reason: str = Field(description="评分理由")


_grader = grade_model.with_structured_output(GradeAnswerResult)


def grade_answer_from_context(state: RagAgentState):
    contexts = state.get("context", [])
    context_text = "\n---\n".join(c["text"] for c in contexts) if contexts else "（无上下文）"

    prompt = f"""你是检查助手,检查回答的答案是否忠诚于上下文。

用户问题:
{state['question']}

检索到的上下文:
{context_text}

生成的答案:
{state.get('answer', '')}

请评估答案是否忠实于上下文,不要编造信息。"""

    result: GradeAnswerResult = _grader.invoke(prompt)

    return {
        "answer_grade": result,
        "trace": [f"grade_answer: score={result.score}, faithful={result.is_faithful}"],
    }
