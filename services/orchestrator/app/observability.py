from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageMetric:
    name: str
    duration_ms: int
    documents_retrieved: int = 0
    source_errors: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class RunObservability:
    run_id: str
    started_at: str = field(default_factory=_now_iso)
    ended_at: str | None = None
    total_duration_ms: int = 0
    total_estimated_cost_usd: float = 0.0
    stage_metrics: list[StageMetric] = field(default_factory=list)
    budget_stop_reason: str | None = None

    def add_stage(self, metric: StageMetric) -> None:
        self.stage_metrics.append(metric)
        self.total_estimated_cost_usd += metric.estimated_cost_usd
        self.total_duration_ms += metric.duration_ms

    def finish(self, budget_stop_reason: str | None = None) -> None:
        self.ended_at = _now_iso()
        self.budget_stop_reason = budget_stop_reason

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "stage_metrics": [asdict(stage) for stage in self.stage_metrics],
        }
