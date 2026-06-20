"""OpenAI-compatible request/response models (Pydantic v2)."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# --- Request ----------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    # OpenAI allows content to be a plain string or a list of content parts.
    content: Union[str, list[dict[str, Any]], None] = None
    name: Optional[str] = None

    def text(self) -> str:
        """Flatten content to plain text regardless of shape."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for part in self.content:
            if isinstance(part, dict):
                # {"type": "text", "text": "..."} style parts
                value = part.get("text") or part.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    # Accepted but ignored — present so PUM's payloads validate cleanly.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    stop: Union[str, list[str], None] = None

    model_config = {"extra": "ignore"}


# --- Response ---------------------------------------------------------------

class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


# --- /v1/models -------------------------------------------------------------

class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "notebooklm-proxy"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
