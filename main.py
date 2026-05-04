import asyncio
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.redis import RedisStore

from rag.rag_chain import rag_search
from sql.api.bookingAPi import router as booking_router
from sql.api.venueApi import router as venue_router
from sql.crud.message import save_to_db
from sql.entity.studentForm import StudentForm
from sql.tools.tool1 import *

# 获取当前模块的日志记录器（名字就是 main）
logger = logging.getLogger(__name__)

# 配置一下基本格式（如果不配置，默认可能看不到输出）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()
print(f"DEBUG: 尝试连接 Redis 地址: {os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}")
app = FastAPI()
app.include_router(venue_router)
app.include_router(booking_router)

model = init_chat_model(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model_provider="openai",
    model="qwen3.5-35b-a3b",
)
extractor_model = init_chat_model(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model_provider="openai",
    model="qwen-turbo",
)

def get_message_text(msg) -> str:
    """安全地提取消息文本，处理 content 是 list 的情况"""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        # 混合内容: 提取所有文本片段
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in msg.content
        )
    return str(msg.content)

def format_messages(messages: list):
    parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "用户"
        elif isinstance(msg, AIMessage):
            role = "AI"
        elif isinstance(msg, ToolMessage):
            role = "工具结果"
        else:
            role = "未知"
        text = get_message_text(msg)
        parts.append(f"[{role}]: {text}")
    return "\n".join(parts)

def extract_relevant_memory(full_history: list, user_question: str) -> str:
    """从完整历史中提取与当前提问相关的记忆"""
    if not user_question:
        return "/n".join(full_history)

    prompt = f"""以下是完整对话历史,用户当前的问题是: "{user_question}"
        请提取与当前问题相关的记忆,只保留有用的上下文,不相关的直接丢弃。
        输出格式为简洁的摘要。
        ---
        对话历史:
        {format_messages(full_history)}
    """
    result = extractor_model.invoke([HumanMessage(prompt)])
    return result.content


tools = [
    rag_search, query_venues, check_availability,
    create_booking, cancel_booking
]
# 模型必须绑定工具，才知道可以输出 tool_calls create_agent方法帮你绑定了工具而已 自定义图的时候就得自己绑定工具
model_with_tools = model.bind_tools(tools)

class AgentState(MessagesState):
    last_summarized_index: int = 0  # 默认0,表示还没有做记忆摘要

def should_summarize(state):
    msgs = state.get("messages", [])
    last_idx = state.get("last_summarized_index", 0)
    return (len(msgs) - last_idx) >= 20  # 新增20条就触发

def summarize_and_store(state,config:RunnableConfig,*,store:BaseStore):
    msgs = state.get("messages",[])
    last_index = state.get("last_summarized_index",0)

    to_summary = msgs[last_index:last_index+10]
    summary = extractor_model.invoke([
        SystemMessage("压缩以下对话为摘要:"),
        HumanMessage(format_messages(to_summary)),  # 包成 HumanMessage
    ]).content
    store.put(
        ("user",config["configurable"]["thread_id"]),
        "summary",
        {"content": summary},
    )
    return {
        "last_summarized_index": last_index+len(to_summary),
    }

def agent_node(state,config:RunnableConfig,*,store:BaseStore):
    """调模型,返回响应"""
    msgs = state["messages"]
    last_idx = state.get("last_summarized_index",0)
    summary_item = store.get(("user", config["configurable"]["thread_id"]), "summary")
    summary = summary_item.value["content"] if summary_item else None
    # 截取未摘要的部分
    unsummarized = msgs[last_idx:]
    if summary:
        context = [HumanMessage(summary)] + unsummarized
    else:
        context = unsummarized

    response = model_with_tools.invoke(context)
    return {"messages": [response]}


def graph_build():
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("summarize",summarize_and_store)
    # 设置入口
    builder.set_entry_point("agent")

    # 条件边
    def should_use_tool(state) -> str:
        """看最后一条消息有没有 tool_calls"""
        last = state['messages'][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    builder.add_conditional_edges(
        "agent",
        should_use_tool,
        {"tools": "tools", "__end__": "__end__"},
    )
    builder.add_conditional_edges(
        "tools",
        should_summarize,
        {True: "summarize", False: "agent"}
    )
    # 工具执行完后，固定回到 agent（让它基于工具结果继续推理）
    builder.add_edge("summarize","agent")
    # Redis 来做记忆持久化
    # "redis://localhost:6379"
    REDIS_URL = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
    redis_checkpointer = RedisSaver(
        redis_url=REDIS_URL,
        ttl={
            "default_ttl": 10080,  # 这个单位是分钟
            "refresh_on_read": False,  # 读取不刷新
        },
    )
    redis_store = RedisStore(
        conn=redis_checkpointer._redis,
    )
    redis_store.setup()  # 初始化
    redis_checkpointer.setup()  # 必须调! 初始化 _key_registry
    graph = builder.compile(checkpointer=redis_checkpointer,store=redis_store)
    return graph

graph = graph_build()
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(studentForm: StudentForm):
    config = {"configurable": {"thread_id": str(studentForm.id)}}
    try:
        response = graph.invoke(
            {"messages": [
                HumanMessage(studentForm.content)
            ]},
            config=config,
        )
        answer = response["messages"][-1].content
        #异步写入Mysql数据库对话消息
        asyncio.create_task(
            save_to_db(studentForm.id,studentForm.content,answer)
        )
        return answer
    except Exception as e:
        # 记录日志,返回友好提示
        logger.error(f"Chat error: {e}")
        return {"error": "服务暂时不可用,请稍后重试"}
