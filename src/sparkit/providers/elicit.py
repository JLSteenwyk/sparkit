from __future__ import annotations

from sparkit.evidence.schema import EvidenceItem
from sparkit.providers.base import ProviderQuery


class ElicitProvider:
    name = "elicit"

    def search(self, query: ProviderQuery) -> list[EvidenceItem]:
        # TODO: Replace with real Elicit integration call(s), subject to official API access.
        _ = query
        return []

