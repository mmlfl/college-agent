from datetime import datetime

from sqlalchemy.orm import Session

from graph.sql.sql_previous_work.table.models import engine, Message


async def save_to_db(student_id: int, question: str, answer: str):
    with Session(engine) as session:
        message = Message(
            student_id=student_id,
            question=question,
            answer=answer,
            create_time=datetime.now(),
        )
        session.add(message)
        session.commit()
