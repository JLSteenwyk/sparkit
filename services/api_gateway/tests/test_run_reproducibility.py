from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from services.api_gateway.app.state import RunStore
from shared.schemas.domain import Constraints, Mode, Run, Status


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL is required for DB-backed tests")
def test_run_reproducibility_roundtrip() -> None:
    store = RunStore(database_url=os.getenv("DATABASE_URL"))

    run_id = f"run_repr_{uuid4().hex}"
    now = datetime.now(timezone.utc)
    run = Run(
        run_id=run_id,
        question="test reproducibility",
        mode=Mode.SINGLE,
        status=Status.QUEUED,
        constraints=Constraints(),
        providers=["openai"],
        prompt_version="prompt_x",
        config_version="config_y",
        reproducibility={"fingerprint": "abc123", "prompt_version": "prompt_x"},
        created_at=now,
        updated_at=now,
    )

    store.create_run(run)
    loaded = store.get_run(run_id)
    assert loaded is not None
    assert loaded.prompt_version == "prompt_x"
    assert loaded.config_version == "config_y"
    assert loaded.reproducibility.get("fingerprint") == "abc123"
