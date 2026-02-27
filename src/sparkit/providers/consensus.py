from __future__ import annotations

from sparkit.evidence.schema import EvidenceItem
from sparkit.providers.base import ProviderQuery


class ConsensusProvider:
    name = "consensus"

    def search(self, query: ProviderQuery) -> list[EvidenceItem]:
        # TODO: Replace with real Consensus integration call(s), subject to official API access.
        _ = query
        return []

