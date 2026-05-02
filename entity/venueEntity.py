from pydantic import BaseModel


# 返回时用的（有 id 字段）
class VenueEntity(BaseModel):
    id: int | None=None
    name: str
    type: str
    location: str

    class Config:
        from_attributes = True