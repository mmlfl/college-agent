from datetime import datetime

from sqlalchemy.orm import Session

from sql.entity.MessageEntity import MessageEntity
from sql.table.models import engine


async def save_to_db(studentId:int,question:str,answer:str):
    with Session(engine) as session :
        message = MessageEntity(
            student_id=studentId,
            question=question,
            answer=answer,
            create_time=datetime.now(),
        )
        session.add(message)
        session.commit()


