from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


DEFAULT_ARROW_PATH = Path(
    "/home/bizon/.cache/huggingface/datasets/futurehouse___hle-gold-bio-chem/default/0.0.0/"
    "1feb9e1d545731dba81e594438330406830e5260/hle-gold-bio-chem-train.arrow"
)


def _load_answer_map(arrow_path: Path) -> dict[str, dict[str, str]]:
    with pa.memory_map(str(arrow_path), "r") as source:
        reader = ipc.RecordBatchStreamReader(source)
        table = reader.read_all()

    answer_map: dict[str, dict[str, str]] = {}
    for idx in range(table.num_rows):
        source_id = str(table["id"][idx].as_py())
        answer = str(table["answer"][idx].as_py() or "")
        answer_type = str(table["answer_type"][idx].as_py() or "")
        answer_map[source_id] = {"answer": answer, "answer_type": answer_type}
    return answer_map


def enrich_questions_file(path: Path, answer_map: dict[str, dict[str, str]]) -> tuple[int, int]:
    rows = json.loads(path.read_text())
    updated = 0
    missing = 0
    for row in rows:
        source_id = str(row.get("source_id", ""))
        mapped = answer_map.get(source_id)
        if not mapped:
            missing += 1
            continue
        row["answer_type"] = mapped["answer_type"] or row.get("answer_type")
        row["correct_answer"] = mapped["answer"]
        updated += 1
    path.write_text(json.dumps(rows, indent=2))
    return updated, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill HLE benchmark question files with gold answers.")
    parser.add_argument(
        "--arrow-path",
        default=str(DEFAULT_ARROW_PATH),
        help="Path to cached HLE arrow file containing `id`, `answer`, and `answer_type` columns.",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=[
            "benchmarks/hle_gold_bio_chem/questions_full.json",
            "benchmarks/hle_gold_bio_chem/questions_bio10_chem10.json",
        ],
        help="Question JSON files to enrich.",
    )
    args = parser.parse_args()

    arrow_path = Path(args.arrow_path)
    answer_map = _load_answer_map(arrow_path)
    for question_path in args.questions:
        path = Path(question_path)
        updated, missing = enrich_questions_file(path, answer_map)
        print(f"{path}: updated={updated} missing_source_ids={missing}")


if __name__ == "__main__":
    main()
