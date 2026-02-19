from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

PROMPT_VERSION = "synthesis_v1.2"
CONFIG_VERSION = "orchestration_v1.2"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_reproducibility_record(
    question: str,
    mode: str,
    providers: list[str],
    constraints: dict,
    prompt_version: str = PROMPT_VERSION,
    config_version: str = CONFIG_VERSION,
) -> dict:
    question_hash = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
    stable_payload = {
        "question_hash": question_hash,
        "mode": mode,
        "providers": sorted(providers),
        "constraints": constraints,
        "prompt_version": prompt_version,
        "config_version": config_version,
        "app_version": os.getenv("APP_VERSION", "0.1.0"),
    }
    payload = {
        **stable_payload,
        "created_at": _utc_now_iso(),
    }
    # Fingerprint is intentionally computed from stable inputs only.
    payload["fingerprint"] = _fingerprint(stable_payload)
    return payload
