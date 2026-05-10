import operator
from typing import TypedDict, Annotated, Required


class RagAgentState(TypedDict, total=False):
    question: Required[str]
    rewritten_questions: list[str]                     # 改写后的问题,可能多条
    context: list[dict]                                 # 精排后top chunks [{score,text,source,doc_type,product_id,product_name}]
    context_grade: dict                                 # GradeResult(is_relevant, score, reason)
    answer_grade: dict                                  # GradeAnswerResult(is_faithful, score, reason)
    answer: str
    iterations: int                                     # 迭代次数(防止无限循环)
    trace: Annotated[list[str], operator.add]           # 执行轨迹(累积)