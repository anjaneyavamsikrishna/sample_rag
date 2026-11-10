"""Request and response shapes. FastAPI validates every request against these and builds /docs from them."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=20,
                                   description="Earlier messages of THIS conversation. The API is stateless: "
                                               "the client sends the history with every call.")
    source: str | None = Field(default=None, max_length=255, description="Restrict the search to one file name.")

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class SourceOut(BaseModel):
    id: int
    source: str
    page: int | None = None
    location: str | None = None
    score: float
    text: str
    cited: bool


class AskResponse(BaseModel):
    question: str
    standalone_question: str
    rewritten: bool
    answer: str
    sources: list[SourceOut]
    abstained: bool
    reason: str | None = None
    mode: Literal["llm", "extractive", "error"]
    warnings: list[str] = []
    best_score: float


class UploadResult(BaseModel):
    status: Literal["indexed", "replaced", "duplicate", "error"]
    source: str
    chunks: int = 0
    doc_id: str | None = None
    detail: str | None = None


class UploadResponse(BaseModel):
    results: list[UploadResult]


class DocumentOut(BaseModel):
    doc_id: str
    source: str
    chunks: int
