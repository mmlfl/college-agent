from pydantic import BaseModel


class StudentForm(BaseModel):
    id:int
    content:str
