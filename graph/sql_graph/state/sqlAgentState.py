from jedi.inference.gradual.typing import TypedDict


class sqlAgentState(TypedDict):
    # 用户输入
    user_query: str
    refined_query: str