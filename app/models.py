from pydantic import BaseModel
from typing import Literal


class ChatRequest(BaseModel):
    session_id: str
    message: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
