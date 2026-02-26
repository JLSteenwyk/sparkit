from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.schemas.api import AskRequest
from shared.schemas.domain import Constraints, Mode


def test_option_graph_v2_requires_providers() -> None:
    with pytest.raises(ValidationError):
        AskRequest(
            question="Which option is correct?",
            mode=Mode.OPTION_GRAPH_V2,
            providers=[],
            constraints=Constraints(min_sources=1, max_cost_usd=3.0),
        )


def test_option_graph_v2_accepts_provider_list() -> None:
    req = AskRequest(
        question="Which option is correct?",
        mode=Mode.OPTION_GRAPH_V2,
        providers=["openai"],
        constraints=Constraints(min_sources=1, max_cost_usd=3.0),
    )
    assert req.mode == Mode.OPTION_GRAPH_V2
    assert req.providers == ["openai"]


def test_simple_rag_requires_providers() -> None:
    with pytest.raises(ValidationError):
        AskRequest(
            question="Which option is correct?",
            mode=Mode.SIMPLE_RAG,
            providers=[],
            constraints=Constraints(min_sources=1, max_cost_usd=3.0),
        )
