from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sparkit.evidence.schema import EvidenceItem


@dataclass(frozen=True)
class ProviderQuery:
    question: str
    max_items: int = 20


class EvidenceProvider(Protocol):
    name: str

    def search(self, query: ProviderQuery) -> list[EvidenceItem]:
        ...

