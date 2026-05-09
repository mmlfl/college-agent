from datetime import datetime

from pydantic import BaseModel


class MessageEntity(BaseModel):
    id:int | None=None
    student_id:int
    question:str
    answer:str
    create_time: datetime

    class Config:
        from_attributes = True