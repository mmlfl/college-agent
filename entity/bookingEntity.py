from datetime import time, datetime

from pydantic import BaseModel

class BookingEntity(BaseModel):
    id:int | None=None
    student_id:int
    venue_id:int
    start_time:datetime
    end_time:datetime
    status:str | None=None

    class Config:
        from_attributes=True