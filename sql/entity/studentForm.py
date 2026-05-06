from typing import Any, Optional

from pydantic import BaseModel


class StudentForm(BaseModel):
    id: int
    content: str
    resume: Optional[Any] = None  # 中断恢复时传入的人类审批结果
