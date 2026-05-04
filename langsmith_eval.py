import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langsmith.evaluation import evaluate

load_dotenv()

judge = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3.6-flash"
)


def _extract_from_trace(outputs: dict) -> tuple:
    """倒序遍历 trace，找到当前轮次的 question/context/answer

    记忆系统会导致 messages 列表堆积多轮数据，所以从后往前找：
    - 倒过来第一条 AI → 当前轮 answer
    - 倒过来第一条 rag_search → 当前轮 context
    - 倒过来第一条 human → 当前轮 question
    """
    if not outputs:
        return "", "", ""

    messages = outputs.get("messages", [])

    question = ""
    context = ""
    answer = ""

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        msg_type = msg.get("type", msg.get("lc", ""))

        if not answer and msg_type == "ai":
            content = msg.get("content", "")
            answer = content if isinstance(content, str) else str(content)
        elif not context and msg_type == "tool" and msg.get("name") == "rag_search":
            content = msg.get("content", "")
            context = content if isinstance(content, str) else str(content)
        elif not question and msg_type == "human":
            content = msg.get("content", "")
            question = content if isinstance(content, str) else str(content)

        # 三个都找到了就提前退出
        if question and context and answer:
            break

    print(f"question: {question[:50]}..., context: {context[:50]}..., answer: {answer[:50]}...")
    return question, context, answer


# rag_predict 什么都不做,只保证 evaluate() 能跑
def rag_predict(inputs: dict) -> dict:
    return {}


# 2. 评估器 -- 不用 exact_match,用 LLM-as-judge

# 答案相关性: 回答是否解决了问题
def answer_relevance(run, example) -> dict:
    """评估已存的回答是否回答了用户的问题"""
    question, context, answer = _extract_from_trace(example.outputs)
    if not answer:
        return {"key": "answer_relevance", "score": 0.0}

    prompt = f"""
用户问题: {question}
AI回答: {answer}
请判断回答是否直接解决了用户的问题。
返回一个 0.0 到 1.0 之间的浮点数(分数越高说明回答越好)。
只返回数字,不要返回其他文字。
"""
    result = judge.invoke([HumanMessage(prompt)])
    try:
        score = float(result.content.strip())
    except (ValueError, AttributeError):
        score = 0.5

    return {"key": "answer_relevance", "score": score}


# 忠实度: 回答是否基于检索到的上下文(无幻觉)
def faithfulness(run, example) -> dict:
    """评估已存的回答是否基于检索到的上下文(无幻觉)"""
    question, context, answer = _extract_from_trace(example.outputs)
    if not context or not answer:
        return {"key": "faithfulness", "score": 0.0}

    prompt = f"""检索到的上下文:{context[:1000]}AI回答:{answer}
请判断回答是否完全基于上面的上下文,没有编造信息。
返回一个 0.0 到 1.0 之间的浮点数(0.0 = 完全编造, 1.0 = 完全忠实,中间值表示部分忠实)。
只返回数字,不要返回其他文字。
"""
    result = judge.invoke([HumanMessage(prompt)])
    try:
        score = float(result.content.strip())
    except (ValueError, AttributeError):
        score = 0.5

    return {"key": "faithfulness", "score": score}


# 上下文相关性: 检索到的内容是否紧扣问题,没有多余噪声
def context_relevance(run, example) -> dict:
    """评估检索到的上下文是否紧密相关于用户问题,没有多余内容"""
    question, context, answer = _extract_from_trace(example.outputs)
    if not question or not context:
        return {"key": "context_relevance", "score": 0.0}

    prompt = f"""用户问题: {question}
检索到的上下文:
{context[:1000]}

请判断检索到的上下文与用户问题的紧密相关程度。
返回 0.0 到 1.0 之间的浮点数,分数越高代表越紧密:
1.0 = 每句话都直接回答问题
0.7~0.9 = 高度相关,少量无关信息
0.4~0.6 = 部分相关,包含较多噪声
0.0~0.3 = 几乎不相关或全是噪声
只返回数字,不要返回其他文字。
"""
    result = judge.invoke([HumanMessage(prompt)])
    try:
        score = float(result.content.strip())
    except (ValueError, AttributeError):
        score = 0.5

    return {"key": "context_relevance", "score": score}


# 3. 运行评估
evaluate(
    rag_predict,
    data="langchian-ai-learning-dataset",
    evaluators=[answer_relevance, faithfulness, context_relevance],
    experiment_prefix="rag-v1-test",
    max_concurrency=3,  # 并发数,别太大避免被限流
)
