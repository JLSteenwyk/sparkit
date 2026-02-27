#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _coerce_json_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        try:
            json.dumps(value)
            cleaned[key] = value
        except TypeError:
            # Skip non-JSON-serializable payloads (e.g. decoded PIL images).
            continue
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FutureHouse HLE-Gold bio/chem dataset from Hugging Face.")
    parser.add_argument("--dataset", default="futurehouse/hle-gold-bio-chem")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="benchmarks/hle_gold/questions_full.json")
    parser.add_argument("--max-questions", type=int, default=0)
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install datasets`."
        ) from exc

    ds = load_dataset(args.dataset, split=args.split)
    rows = [_coerce_json_row(dict(item)) for item in ds]
    if args.max_questions > 0:
        rows = rows[: args.max_questions]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    main()
