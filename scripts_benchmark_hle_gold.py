#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from sparkit.orchestrator.answering import (
    decide_mcq_fast_consensus,
    decide_mcq_from_evidence,
    decide_mcq_nano_consensus10,
    parse_mcq,
)
from sparkit.orchestrator.federation import FederationConfig, build_evidence_pack
from sparkit.providers.consensus import ConsensusProvider
from sparkit.providers.elicit import ElicitProvider
from sparkit.providers.exa import ExaProvider
from sparkit.providers.paperqa2 import PaperQA2Provider
from sparkit.providers.scite import SciteProvider


def _provider_factory(name: str):
    mapping = {
        "paperqa2": PaperQA2Provider,
        "exa": ExaProvider,
        "elicit": ElicitProvider,
        "consensus": ConsensusProvider,
        "scite": SciteProvider,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported provider: {name}")
    return mapping[name]()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark federated SPARKIT MVP on HLE-Gold MCQ questions.")
    parser.add_argument("--questions", default="benchmarks/hle_gold/questions_full.json")
    parser.add_argument("--providers", default="paperqa2,exa,elicit,consensus,scite")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--provider-max-items", type=int, default=20)
    parser.add_argument("--enable-mcq-option-queries", action="store_true")
    parser.add_argument("--max-query-variants", type=int, default=6)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--label", default="hle_gold_mvp")
    parser.add_argument("--decision-mode", choices=["llm", "fast_consensus", "nano_consensus10"], default="llm")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    if args.max_questions > 0:
        questions = questions[: args.max_questions]

    providers = [_provider_factory(name.strip()) for name in args.providers.split(",") if name.strip()]
    cfg = FederationConfig(
        top_k=args.top_k,
        provider_max_items=args.provider_max_items,
        enable_mcq_option_queries=bool(args.enable_mcq_option_queries),
        max_query_variants=args.max_query_variants,
    )

    predictions = []
    correct = 0
    scored = 0
    t0 = time.monotonic()

    for row in questions:
        qid = str(row.get("id"))
        question = str(row.get("question", ""))
        answer_type = str(row.get("answer_type", "")).strip().lower()
        raw_answer = row.get("correct_answer")
        if raw_answer is None:
            raw_answer = row.get("answer")
        correct_answer = _extract_correct_letter(raw_answer)

        _stem, choices = parse_mcq(question)
        if answer_type != "multiplechoice" or not choices or not correct_answer:
            predictions.append(
                {
                    "id": qid,
                    "answer_text": "",
                    "answer_confidence": 0.0,
                    "status": "skipped_non_mcq_or_missing_key",
                }
            )
            continue

        pack = build_evidence_pack(question=question, providers=providers, config=cfg)
        if args.decision_mode == "fast_consensus":
            decision = decide_mcq_fast_consensus(question, pack)
        elif args.decision_mode == "nano_consensus10":
            decision = decide_mcq_nano_consensus10(question, pack)
        else:
            decision = decide_mcq_from_evidence(question, pack)
        answer_text = f"<answer>{decision.answer_letter}</answer>" if decision.answer_letter else ""
        is_correct = 1 if (decision.answer_letter or "") == correct_answer else 0
        scored += 1
        correct += is_correct

        predictions.append(
            {
                "id": qid,
                "answer_text": answer_text,
                "answer_confidence": decision.confidence,
                "is_correct": is_correct,
                "correct_answer": correct_answer,
                "provider_stats": pack.provider_stats,
                "rationale": decision.rationale,
            }
        )

    avg_score = (correct / scored) if scored else 0.0
    runtime_seconds = round(time.monotonic() - t0, 3)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.label}_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "num_questions": len(questions),
                "num_scored_mcq": scored,
                "correct": correct,
                "average_score": avg_score,
                "runtime_seconds": runtime_seconds,
                "providers": [provider.name for provider in providers],
                "decision_mode": args.decision_mode,
                "config": asdict(cfg),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "average_score": avg_score,
                "correct": correct,
                "scored": scored,
                "runtime_seconds": runtime_seconds,
                "decision_mode": args.decision_mode,
            },
            indent=2,
        )
    )


def _extract_correct_letter(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if not text:
        return ""
    first = text[:1]
    return first if "A" <= first <= "Z" else ""


if __name__ == "__main__":
    main()
