import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore

from sql.booking.bookingAPi import router as booking_router
from sql.booking.venueApi import router as venue_router
from sql.entity.studentForm import StudentForm

from rag.rag_chain import rag_search
from sql.tools.tool1 import *

load_dotenv()
app = FastAPI()
app.include_router(venue_router)
app.include_router(booking_router)

model = init_chat_model(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model_provider="openai",
    model="qwen3.5-35b-a3b",
)
tools = [
    rag_search, query_venues, check_availability,
    create_booking, cancel_booking
]
# 模型必须绑定工具，才知道可以输出 tool_calls create_agent方法帮你绑定了工具而已 自定义图的时候就得自己绑定工具
model_with_tools = model.bind_tools(tools)

builder = StateGraph(MessagesState)

def agent_node(state):
    """调模型,返回响应"""
    response = model_with_tools.invoke(state['messages'])
    return {"messages": [response]}

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
#设置入口
builder.set_entry_point("agent")
#条件边
def should_use_tool(state) -> str:
    """看最后一条消息有没有 tool_calls"""
    last = state['messages'][-1]
    if isinstance(last,AIMessage) and last.tool_calls:
        return "tools"
    return "__end__"
builder.add_conditional_edges(
    "agent",
    should_use_tool,
    {"tools":"tools","__end__":"__end__"},
)
# 工具执行完后，固定回到 agent（让它基于工具结果继续推理）
builder.add_edge("tools", "agent")

# Redis 来做记忆持久化
#"redis://localhost:6379"
REDIS_URL = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
redis_checkpointer = RedisSaver(redis_url=REDIS_URL)
redis_checkpointer.setup()
graph = builder.compile(checkpointer=redis_checkpointer)


agent = create_agent(
    model=model,
    tools=[
        rag_search,
        query_venues,
        check_availability,
        create_booking,
        cancel_booking,
    ],
    checkpointer=redis_checkpointer,
)
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(studentForm:StudentForm):
    config = {"configurable":{"thread_id":{studentForm.id}}}
    response = graph.invoke(
        {"messages":[
            HumanMessage(studentForm.content)
        ]},
        config=config,
    )
    return response["messages"][-1].content


