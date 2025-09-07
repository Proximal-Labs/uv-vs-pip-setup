
from pydantic import BaseModel
from typing import Optional

class TodoCreate(BaseModel):
    title: str
    description: Optional[str] = None
    status: Optional[str] = "pending"

class TodoUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None

class TodoOut(TodoCreate):
    id: int
    class Config:
        orm_mode = True
