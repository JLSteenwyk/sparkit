#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx


@dataclass
class SmokeResult:
    provider: str
    model: str
    attempted_payload: dict[str, object]
    ok: bool
    status_code: int | None
    latency_s: float
    answer_excerpt: str
    error: str | None


def _now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _excerpt(text: str, limit: int = 240) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _openai_smoke(prompt: str) -> SmokeResult:
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    payload: dict[str, object] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 280,
        "reasoning": {"effort": "xhigh"},
    }
    if not api_key:
        return SmokeResult("openai", model, payload, False, None, 0.0, "", "OPENAI_API_KEY missing")
    started = perf_counter()
    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
        latency = max(0.0, perf_counter() - started)
        data = resp.json() if resp.content else {}
        text = str(data.get("output_text", "") or "")
        if not text.strip():
            for item in data.get("output", []) or []:
                if item.get("type") != "message":
                    continue
                for content in item.get("content", []) or []:
                    if content.get("type") == "output_text":
                        text += str(content.get("text", "") or "")
        return SmokeResult(
            provider="openai",
            model=model,
            attempted_payload=payload,
            ok=resp.is_success,
            status_code=resp.status_code,
            latency_s=latency,
            answer_excerpt=_excerpt(text),
            error=None if resp.is_success else _excerpt(resp.text, 500),
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeResult("openai", model, payload, False, None, max(0.0, perf_counter() - started), "", str(exc))


def _anthropic_smoke(prompt: str) -> SmokeResult:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
    payload: dict[str, object] = {
        "model": model,
        "max_tokens": 3200,
        "messages": [{"role": "user", "content": prompt}],
        "thinking": {"type": "enabled", "budget_tokens": 3000},
    }
    if not api_key:
        return SmokeResult("anthropic", model, payload, False, None, 0.0, "", "ANTHROPIC_API_KEY missing")
    started = perf_counter()
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
        latency = max(0.0, perf_counter() - started)
        data = resp.json() if resp.content else {}
        text = ""
        for block in data.get("content", []) or []:
            if block.get("type") == "text":
                text += str(block.get("text", "") or "")
        return SmokeResult(
            provider="anthropic",
            model=model,
            attempted_payload=payload,
            ok=resp.is_success,
            status_code=resp.status_code,
            latency_s=latency,
            answer_excerpt=_excerpt(text),
            error=None if resp.is_success else _excerpt(resp.text, 500),
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeResult(
            "anthropic",
            model,
            payload,
            False,
            None,
            max(0.0, perf_counter() - started),
            "",
            str(exc),
        )


def _gemini_smoke(prompt: str) -> SmokeResult:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    model = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
    payload: dict[str, object] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 3200,
            "thinkingConfig": {"thinkingBudget": 24576},
        },
    }
    if not api_key:
        return SmokeResult("gemini", model, payload, False, None, 0.0, "", "GEMINI_API_KEY/GOOGLE_API_KEY missing")
    started = perf_counter()
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                json=payload,
            )
        latency = max(0.0, perf_counter() - started)
        data = resp.json() if resp.content else {}
        text = ""
        candidates = data.get("candidates", []) or []
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", []) or []
            text = "".join(str(part.get("text", "") or "") for part in parts)
        return SmokeResult(
            provider="gemini",
            model=model,
            attempted_payload=payload,
            ok=resp.is_success,
            status_code=resp.status_code,
            latency_s=latency,
            answer_excerpt=_excerpt(text),
            error=None if resp.is_success else _excerpt(resp.text, 500),
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeResult(
            "gemini",
            model,
            payload,
            False,
            None,
            max(0.0, perf_counter() - started),
            "",
            str(exc),
        )


def main() -> int:
    prompt = (
        "Smoke test. Answer briefly: Which molecule stores hereditary information in nearly all known cellular life?"
    )
    results = [
        _openai_smoke(prompt),
        _anthropic_smoke(prompt),
        _gemini_smoke(prompt),
    ]
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"reasoning_smoke_{_now_slug()}.json"
    out_path.write_text(json.dumps([asdict(item) for item in results], indent=2))
    print(out_path)
    for item in results:
        print(
            f"{item.provider:10s} ok={item.ok} status={item.status_code} "
            f"latency_s={item.latency_s:.2f} model={item.model}"
        )
        if item.error:
            print(f"  error: {item.error}")
        if item.answer_excerpt:
            print(f"  answer: {item.answer_excerpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
