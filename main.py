import os

from fastapi import FastAPI
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from booking.bookingAPi import router as booking_router
from booking.venueApi import router as venue_router
from entity.studentForm import StudentForm

from rag.rag_chain import rag_search

app = FastAPI()
app.include_router(venue_router)
app.include_router(booking_router)

model = init_chat_model(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model_provider="openai",
    model="qwen3.5-35b-a3b"
)
agent = create_agent(
    model=model,
    tools=[rag_search],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(studentForm:StudentForm):
    response = agent.invoke({
        "messages":[
            HumanMessage(studentForm.content)
        ]
    })
    return response["messages"][-1].content
