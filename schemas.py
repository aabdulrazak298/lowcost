"""Pydantic models for request validation."""
from typing import Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ToolFunction(BaseModel):
    name: str
    description: str = ""
    parameters: dict = Field(default_factory=dict)


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1, le=131072)
    stream: bool = False
    tools: list[Tool] | None = None
    tool_choice: str | dict | None = None
    model: str | None = None
