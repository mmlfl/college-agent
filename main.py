import asyncio
import logging

from fastapi import FastAPI
from langchain_core.messages import HumanMessage

from graph import graph
from sql.api.bookingAPi import router as booking_router
from sql.api.venueApi import router as venue_router
from sql.crud.message import save_to_db
from sql.entity.studentForm import StudentForm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI()
app.include_router(venue_router)
app.include_router(booking_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat")
async def chat(studentForm: StudentForm):
    config = {"configurable": {"thread_id": str(studentForm.id)}}
    try:
        response = graph.invoke(
            {"messages": [HumanMessage(studentForm.content)]},
            config=config,
        )
        answer = response["messages"][-1].content
        asyncio.create_task(
            save_to_db(studentForm.id, studentForm.content, answer)
        )
        return answer
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"error": "服务暂时不可用,请稍后重试"}
