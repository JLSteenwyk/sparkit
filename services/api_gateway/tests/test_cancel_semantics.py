from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from services.api_gateway.app.state import RunStore
from shared.schemas.domain import Constraints, Mode, Run, Status


def _database_url() -> str:
    return os.getenv("DATABASE_URL", "")


@pytest.mark.skipif(not _database_url(), reason="DATABASE_URL is required for DB-backed tests")
def test_cancel_semantics_terminal_and_missing() -> None:
    store = RunStore(database_url=_database_url())

    run_id = f"run_test_{uuid4().hex}"
    now = datetime.now(timezone.utc)
    run = Run(
        run_id=run_id,
        question="test question",
        mode=Mode.SINGLE,
        status=Status.QUEUED,
        constraints=Constraints(),
        providers=["openai"],
        created_at=now,
        updated_at=now,
    )
    store.create_run(run)

    first = store.cancel_run(run_id)
    assert first == "cancelled"

    store._update_run_fields(  # noqa: SLF001 - test setup on persisted row
        run_id, {"status": Status.COMPLETED.value, "updated_at": datetime.now(timezone.utc)}
    )
    terminal = store.cancel_run(run_id)
    assert terminal == "terminal"

    missing = store.cancel_run(f"run_missing_{uuid4().hex}")
    assert missing == "not_found"
