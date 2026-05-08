import asyncio
import json
import logging

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from starlette.responses import StreamingResponse

from graph import graph, redis_checkpointer, redis_store
from langgraph.types import Command

from graph.ragGraph import rag_graph_builder
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
        # 新会话：清除该 thread 的历史消息和摘要
        if studentForm.new_session:
            thread_id = str(studentForm.id)
            logger.info(f"Clearing session for thread_id={thread_id}")
            redis_checkpointer.delete_thread(thread_id)
            redis_store.delete(("user", thread_id), "summary")
            # 清除后当作新会话处理，不走 resume
            input_data = {"messages": [HumanMessage(studentForm.content)]}
        else:
            # 严格判空：只有非空字符串才算 resume
            is_resume = studentForm.resume is not None and str(studentForm.resume).strip() != ""

            if is_resume:
                logger.info(f"Resuming with: {studentForm.resume}")
                input_data = Command(resume=studentForm.resume)
            else:
                input_data = {"messages": [HumanMessage(studentForm.content)]}

        answer = None
        interrupt_data = None

        for chunk in graph.stream(input_data, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                interrupt_info = chunk["__interrupt__"][0]
                interrupt_data = interrupt_info.value
                logger.info(f"Caught interrupt: {interrupt_data}")
                break
            for node_name, node_output in chunk.items():
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                            answer = msg.content if isinstance(msg.content, str) else str(msg.content)

        if interrupt_data is not None:
            return {"interrupt": True, "data": interrupt_data}

        if answer:
            return answer
        return {"error": "未获取到AI回复"}

    except Exception as e:
        logger.error(f"Chat error: {type(e).__name__}: {e}")
        return {"error": "服务暂时不可用,请稍后重试"}

@app.post("/rag/chat")
async def ragChat(studentForm: StudentForm):
    config = {"configurable": {"thread_id": str(studentForm.id)}}



#TODO Stream_chat 功能目前尚待开发 我的graph是循环流 目前无法使用stream流来逐一打印最终输出
@app.post("/stream/chat")
async def streamChat(studentForm: StudentForm):
    config = {"configurable": {"thread_id": str(studentForm.id)}}
    try:
        def generate():
            # "messages" 模式: event 是 dict, data 字段才是 (message, metadata) 元组
            # 使用 stream 而非 astream，因为 RedisSaver 只实现了同步版本
            for event in graph.stream(
                    {"messages": [HumanMessage(studentForm.content)]},
                    config=config,
                    stream_mode="messages",
                    version="v2",
            ):
                # event = {'type': 'messages', 'ns': (), 'data': (msg, metadata)}
                msg, metadata = event["data"]
                # 只要 AI 回复的文本增量，不要工具调用
                if isinstance(msg.content, str) and msg.content and not msg.tool_calls:
                    yield f"data: {json.dumps({'content': msg.content})}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"error": "服务暂时不可用,请稍后重试"}
