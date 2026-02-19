from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .domain import Constraints, Mode, Status


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Mode
    providers: list[str] | None = None
    constraints: Constraints = Field(default_factory=Constraints)
    answer_style: str = "technical"
    include_trace: bool = False
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "AskRequest":
        if self.mode == Mode.SINGLE and not self.providers:
            raise ValueError("providers is required when mode=single")
        return self


class AskAcceptedResponse(BaseModel):
    run_id: str
    status: Status = Status.QUEUED
    created_at: datetime
    poll_url: str


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorInfo
