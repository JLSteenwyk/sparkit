from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Mode(str, Enum):
    SINGLE = "single"
    ROUTED = "routed"
    ENSEMBLE = "ensemble"


class Status(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CANCELLING = "cancelling"


class Constraints(BaseModel):
    max_latency_s: int = Field(default=120, ge=10, le=600)
    max_cost_usd: float = Field(default=3.0, ge=0.05, le=100.0)
    min_sources: int = Field(default=5, ge=1, le=50)


class Progress(BaseModel):
    stage: str
    percent: int = Field(ge=0, le=100)


class Usage(BaseModel):
    cost_usd: float = 0.0
    latency_s: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0


class ClaimConfidence(BaseModel):
    claim_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class Answer(BaseModel):
    final_text: str
    answer_confidence: float = Field(ge=0.0, le=1.0)
    claim_confidences: list[ClaimConfidence] = Field(default_factory=list)
    uncertainty_reasons: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    claim_id: str
    doc_id: str
    passage_id: str


class Run(BaseModel):
    run_id: str
    question: str
    mode: Mode
    status: Status
    constraints: Constraints
    answer_style: str = "technical"
    providers: list[str] = Field(default_factory=list)
    include_trace: bool = False
    prompt_version: str = "synthesis_v1.2"
    config_version: str = "orchestration_v1.2"
    reproducibility: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunStatusResponse(BaseModel):
    run_id: str
    status: Status
    mode: Mode
    progress: Progress | None = None
    answer: Answer | None = None
    citations: list[Citation] | None = None
    usage: Usage
    created_at: datetime
    updated_at: datetime


class TraceStage(BaseModel):
    name: str
    status: Status
    model: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)


class ProviderUsage(BaseModel):
    provider: str
    model: str
    tokens_input: int
    tokens_output: int
    cost_usd: float


class QualityGates(BaseModel):
    citation_coverage: float = 0.0
    unsupported_claims: int = 0
    contradiction_flags: int = 0


class RunTraceResponse(BaseModel):
    run_id: str
    stages: list[TraceStage]
    provider_usage: list[ProviderUsage]
    quality_gates: QualityGates
