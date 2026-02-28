#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from sparkit.orchestrator.answering import parse_mcq

_ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.IGNORECASE)


def _extract_letter(text: str) -> str:
    m = _ANSWER_TAG_RE.search(text or "")
    return m.group(1).upper() if m else ""


def _extract_correct_letter(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if not text:
        return ""
    first = text[:1]
    return first if "A" <= first <= "Z" else ""


def _prompt(question: str) -> str:
    return (
        "Answer the multiple-choice STEM question.\n"
        "Return ONLY one XML tag in this exact format: <answer>X</answer> where X is one capital letter.\n"
        "No other text.\n\n"
        f"{question}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct OpenAI benchmark on HLE-Gold MCQs.")
    parser.add_argument("--questions", default="benchmarks/hle_gold/questions_full.json")
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--model", default=os.getenv("DIRECT_OPENAI_MODEL", "gpt-4-0125-preview"))
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--label", default="hle_direct_openai_same_model")
    args = parser.parse_args()

    rows = json.loads(Path(args.questions).read_text())
    if args.max_questions > 0:
        rows = rows[: args.max_questions]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    predictions: list[dict] = []
    scored = 0
    correct = 0

    for row in rows:
        qid = str(row.get("id"))
        question = str(row.get("question", ""))
        answer_type = str(row.get("answer_type", "")).strip().lower()
        answer_key = _extract_correct_letter(row.get("answer"))

        _stem, choices = parse_mcq(question)
        if answer_type != "multiplechoice" or not choices or not answer_key:
            predictions.append(
                {
                    "id": qid,
                    "answer_text": "",
                    "answer_confidence": 0.0,
                    "status": "skipped_non_mcq_or_missing_key",
                }
            )
            continue

        answer_text = ""
        confidence = 0.0
        error = None
        try:
            kwargs = {
                "model": args.model,
                "messages": [{"role": "user", "content": _prompt(question)}],
                "temperature": 0,
            }
            # gpt-5* models reject max_tokens and require max_completion_tokens.
            if args.model.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = 12
            else:
                kwargs["max_tokens"] = 12

            resp = client.chat.completions.create(**kwargs)
            answer_text = (resp.choices[0].message.content or "").strip()
            confidence = 0.7 if answer_text else 0.0
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        picked = _extract_letter(answer_text)
        is_correct = 1 if picked == answer_key else 0
        scored += 1
        correct += is_correct
        predictions.append(
            {
                "id": qid,
                "answer_text": answer_text,
                "answer_confidence": confidence,
                "is_correct": is_correct,
                "correct_answer": answer_key,
                "error": error,
            }
        )

    avg = (correct / scored) if scored else 0.0
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.label}_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "num_questions": len(rows),
                "num_scored_mcq": scored,
                "correct": correct,
                "average_score": avg,
                "model": args.model,
            },
            indent=2,
        )
    )
    print(json.dumps({"output_dir": str(out_dir), "average_score": avg, "correct": correct, "scored": scored}, indent=2))


if __name__ == "__main__":
    main()
