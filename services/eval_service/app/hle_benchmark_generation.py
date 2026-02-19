from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "when",
    "which",
    "what",
    "will",
    "were",
    "have",
    "has",
    "been",
    "being",
    "your",
    "their",
    "about",
    "across",
    "between",
    "under",
    "using",
    "used",
    "given",
    "following",
    "among",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())


def _extract_keywords(question: str, max_keywords: int = 4) -> list[str]:
    unique: list[str] = []
    for token in _tokenize(question):
        if token in _STOPWORDS or token.isdigit():
            continue
        if token not in unique:
            unique.append(token)

    scored = sorted(unique, key=lambda tok: (-len(tok), tok))
    selected = scored[:max_keywords]
    if not selected:
        return ["analysis"]
    return selected


def _normalize_domain(category: str) -> str:
    lowered = (category or "").strip().lower()
    if lowered == "chemistry":
        return "chemistry"
    if "biology" in lowered or "medicine" in lowered:
        return "biology"
    return "general"


def _slugify(value: str) -> str:
    lowered = (value or "general").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "general"


def _sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"Requested {count} rows but only {len(rows)} available")
    return rng.sample(rows, count)


def build_hle_biochem_subset(
    rows: list[dict[str, Any]],
    *,
    bio_count: int = 10,
    chem_count: int = 10,
    seed: int = 7,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    bio_rows = [row for row in rows if _normalize_domain(str(row.get("category", ""))) == "biology"]
    chem_rows = [row for row in rows if _normalize_domain(str(row.get("category", ""))) == "chemistry"]

    selected_bio = _sample_rows(bio_rows, bio_count, rng)
    selected_chem = _sample_rows(chem_rows, chem_count, rng)

    # Interleave domains for a balanced run order.
    interleaved: list[dict[str, Any]] = []
    for idx in range(max(len(selected_bio), len(selected_chem))):
        if idx < len(selected_bio):
            interleaved.append(selected_bio[idx])
        if idx < len(selected_chem):
            interleaved.append(selected_chem[idx])

    benchmark_questions: list[dict[str, Any]] = []
    for idx, row in enumerate(interleaved, start=1):
        question_text = str(row.get("question", "")).strip()
        category = str(row.get("category", ""))
        raw_subject = str(row.get("raw_subject", "general"))
        domain = _normalize_domain(category)
        required = _extract_keywords(question_text, max_keywords=4)
        optional = _extract_keywords(raw_subject, max_keywords=3)
        benchmark_questions.append(
            {
                "id": f"q{idx:03d}",
                "question": question_text,
                "domain": domain,
                "subdomain": _slugify(raw_subject),
                "required_keywords": required,
                "optional_keywords": optional,
                "must_have_citations": 2,
                "difficulty": "hard",
                "source_dataset": "futurehouse/hle-gold-bio-chem",
                "source_id": row.get("id"),
                "source_category": category,
                "source_subject": raw_subject,
                "answer_type": row.get("answer_type"),
            }
        )
    return benchmark_questions


def write_hle_biochem_subset(
    path: str | Path,
    *,
    dataset_name: str = "futurehouse/hle-gold-bio-chem",
    bio_count: int = 10,
    chem_count: int = 10,
    seed: int = 7,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Missing optional dependency 'datasets'. Install with `pip install datasets`.") from exc

    dataset = load_dataset(dataset_name)
    train = dataset["train"]
    rows = [dict(train[i]) for i in range(len(train))]

    questions = build_hle_biochem_subset(rows, bio_count=bio_count, chem_count=chem_count, seed=seed)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(questions, indent=2))
    return len(questions)

