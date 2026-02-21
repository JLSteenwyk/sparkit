from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


RESULTS_DIR = Path("benchmarks/results")
OUTPUT_PATH = Path("docs/hle-gold-live-status.md")
QUESTIONS_PATH = Path("benchmarks/hle_gold_bio_chem/questions_full.json")


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _model_for(method: str, run_slug: str) -> str:
    if method == "single_openai":
        return "gpt-5.2"
    if method == "single_openai_pro":
        return "gpt-5.2-pro"
    if method == "single_anthropic":
        return "claude-opus-4-6"
    if method == "single_anthropic_sonnet":
        return "claude-sonnet-4-6"
    if method == "single_gemini":
        return "gemini-3.1-pro-preview" if "gemini31" in run_slug else "gemini-3-pro-preview"
    if method == "single_kimi":
        return "kimi-k2-turbo-preview"
    if method == "single_deepseek":
        return "deepseek-reasoner"
    if method == "single_grok":
        return "grok-4-0709"
    if method == "single_mistral":
        return "mistral-large-2512"
    if method == "direct_openai":
        return "gpt-5.2-pro" if "openai_pro" in run_slug else "gpt-5.2"
    if method == "direct_anthropic":
        return "claude-sonnet-4-6" if "sonnet" in run_slug else "claude-opus-4-6"
    if method == "direct_gemini":
        return "gemini-3.1-pro-preview" if "gemini31" in run_slug else "gemini-3-pro-preview"
    if method == "direct_kimi":
        return "kimi-k2-turbo-preview"
    if method == "direct_deepseek":
        return "deepseek-reasoner"
    if method == "direct_mistral":
        return "mistral-large-2512"
    if method == "direct_grok":
        if "fast_nonreason" in run_slug or "split_v2" in run_slug:
            return "grok-4-fast-non-reasoning"
        if "fast_reason" in run_slug:
            return "grok-4-fast-reasoning"
        return "grok-4-0709"
    return ""


def _load_rows() -> list[dict[str, object]]:
    category_by_id = _question_categories()
    rows: list[dict[str, object]] = []
    for manifest in sorted(RESULTS_DIR.glob("hle_gold_*/manifest.json")):
        run_dir = manifest.parent
        run_slug = run_dir.name
        if "smoke" in run_slug:
            continue
        payload = json.loads(manifest.read_text())
        providers = payload.get("providers") or []
        configs = payload.get("configs") or []

        for provider_row in providers:
            method = f"direct_{provider_row.get('provider')}"
            report_rel = provider_row.get("report_file")
            report_path = run_dir / report_rel if isinstance(report_rel, str) else None
            bio_med, chemistry = _category_scores(report_path, category_by_id)
            rows.append(
                {
                    "run": run_slug,
                    "method": method,
                    "model": _model_for(method, run_slug),
                    "status": provider_row.get("status", "completed"),
                    "n": provider_row.get("num_predictions"),
                    "rubric": provider_row.get("average_rubric_score"),
                    "brier": provider_row.get("brier_score"),
                    "ece": provider_row.get("ece"),
                    "avg_cost": provider_row.get("avg_cost_usd"),
                    "total_cost": provider_row.get("total_cost_usd"),
                    "failures": provider_row.get("failure_count"),
                    "bio_med_rubric": bio_med,
                    "chem_rubric": chemistry,
                    "chem_minus_bio": (chemistry - bio_med) if bio_med is not None and chemistry is not None else None,
                }
            )

        for config_row in configs:
            method = str(config_row.get("name", ""))
            report_rel = config_row.get("report_file")
            report_path = run_dir / report_rel if isinstance(report_rel, str) else None
            bio_med, chemistry = _category_scores(report_path, category_by_id)
            rows.append(
                {
                    "run": run_slug,
                    "method": method,
                    "model": _model_for(method, run_slug),
                    "status": config_row.get("status", "completed"),
                    "n": config_row.get("num_predictions"),
                    "rubric": config_row.get("average_rubric_score"),
                    "brier": config_row.get("brier_score"),
                    "ece": config_row.get("ece"),
                    "avg_cost": config_row.get("avg_cost_usd"),
                    "total_cost": config_row.get("total_cost_usd"),
                    "failures": config_row.get("failure_count"),
                    "bio_med_rubric": bio_med,
                    "chem_rubric": chemistry,
                    "chem_minus_bio": (chemistry - bio_med) if bio_med is not None and chemistry is not None else None,
                }
            )

    known_in_progress = {
        "hle_gold_single_core_v2_20260221T054301Z": [
            ("single_openai", "gpt-5.2"),
            ("single_anthropic", "claude-opus-4-6"),
            ("single_gemini", "gemini-3-pro-preview"),
            ("single_kimi", "kimi-k2-turbo-preview"),
            ("single_deepseek", "deepseek-reasoner"),
            ("single_mistral", "mistral-large-2512"),
            ("single_grok", "grok-4-0709"),
        ],
        "hle_gold_single_overrides_v2_20260221T054301Z": [
            ("single_openai_pro", "gpt-5.2-pro"),
            ("single_anthropic_sonnet", "claude-sonnet-4-6"),
        ],
        "hle_gold_direct_openai_pro_v3_20260221T060305Z": [("direct_openai", "gpt-5.2-pro")],
    }
    active_labels = _active_labels()
    existing = {(str(row["run"]), str(row["method"])) for row in rows}
    for run_slug, methods in known_in_progress.items():
        if not (RESULTS_DIR / run_slug).exists():
            continue
        run_label = run_slug.rsplit("_", 1)[0]
        if run_label not in active_labels:
            continue
        for method, model in methods:
            key = (run_slug, method)
            if key in existing:
                continue
            rows.append(
                {
                    "run": run_slug,
                    "method": method,
                    "model": model,
                    "status": "running",
                    "n": None,
                    "rubric": None,
                    "brier": None,
                    "ece": None,
                    "avg_cost": None,
                    "total_cost": None,
                    "failures": None,
                    "bio_med_rubric": None,
                    "chem_rubric": None,
                    "chem_minus_bio": None,
                }
            )

    rows.sort(
        key=lambda row: (
            0 if row["status"] != "completed" else 1,
            -(row["rubric"] if isinstance(row["rubric"], (int, float)) else -1.0),
            str(row["run"]),
            str(row["method"]),
        )
    )
    return rows


def _question_categories() -> dict[str, str]:
    if not QUESTIONS_PATH.exists():
        return {}
    try:
        questions = json.loads(QUESTIONS_PATH.read_text())
    except Exception:
        return {}
    mapping: dict[str, str] = {}
    for row in questions:
        qid = str(row.get("id", ""))
        category = str(row.get("source_category", "")).strip()
        if not qid:
            continue
        if category.lower() in {"biology/medicine", "biology", "medicine"}:
            mapping[qid] = "biology_medicine"
        elif category.lower() == "chemistry":
            mapping[qid] = "chemistry"
    return mapping


def _category_scores(report_path: Path | None, category_by_id: dict[str, str]) -> tuple[float | None, float | None]:
    if report_path is None or not report_path.exists():
        return None, None
    try:
        report = json.loads(report_path.read_text())
    except Exception:
        return None, None
    rubric_scores = report.get("rubric_scores", [])
    if not isinstance(rubric_scores, list):
        return None, None

    bio_vals: list[float] = []
    chem_vals: list[float] = []
    for item in rubric_scores:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id", ""))
        if not qid:
            continue
        score_raw = item.get("total_score")
        if not isinstance(score_raw, (int, float)):
            continue
        category = category_by_id.get(qid)
        if category == "biology_medicine":
            bio_vals.append(float(score_raw))
        elif category == "chemistry":
            chem_vals.append(float(score_raw))

    bio_avg = (sum(bio_vals) / len(bio_vals)) if bio_vals else None
    chem_avg = (sum(chem_vals) / len(chem_vals)) if chem_vals else None
    return bio_avg, chem_avg


def _active_labels() -> set[str]:
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "cmd"],
            text=True,
        )
    except Exception:
        return set()
    labels: set[str] = set()
    marker = "--label "
    for line in output.splitlines():
        if "scripts_capture_baselines.py" not in line and "scripts_capture_direct_baselines.py" not in line:
            continue
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1].strip()
        label = tail.split()[0].strip("'\"")
        if label:
            labels.add(label)
    return labels


def _render(rows: list[dict[str, object]]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# HLE Gold Live Status",
        "",
        f"Last updated (UTC): {now}",
        "",
        "| Run | Method | Model | Status | N | Avg Rubric | Brier | ECE | Avg Cost/Q ($) | Total Cost ($) | Failures | Bio/Med Rubric | Chem Rubric | Chem-Bio Δ |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run']}`",
                    f"`{row['method']}`",
                    f"`{row['model']}`" if row["model"] else "",
                    str(row["status"]),
                    _fmt(row["n"]),
                    _fmt(row["rubric"]),
                    _fmt(row["brier"]),
                    _fmt(row["ece"]),
                    _fmt(row["avg_cost"]),
                    _fmt(row["total_cost"]),
                    _fmt(row["failures"]),
                    _fmt(row["bio_med_rubric"]),
                    _fmt(row["chem_rubric"]),
                    _fmt(row["chem_minus_bio"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Refresh command: `./venv/bin/python scripts_update_hle_live_status.py`")
    return "\n".join(lines)


def main() -> None:
    rows = _load_rows()
    OUTPUT_PATH.write_text(_render(rows))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
