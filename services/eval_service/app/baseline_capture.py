from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runner import run_benchmark_with_predictions


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    mode: str
    providers: list[str]
    model_overrides: dict[str, str] | None = None


DEFAULT_CONFIGS = [
    BaselineConfig(name="single_openai", mode="single", providers=["openai"]),
    BaselineConfig(
        name="single_openai_pro",
        mode="single",
        providers=["openai"],
        model_overrides={"OPENAI_MODEL": "gpt-5.2-pro"},
    ),
    BaselineConfig(name="single_anthropic", mode="single", providers=["anthropic"]),
    BaselineConfig(
        name="single_anthropic_sonnet",
        mode="single",
        providers=["anthropic"],
        model_overrides={"ANTHROPIC_MODEL": "claude-sonnet-4-6"},
    ),
    BaselineConfig(name="single_gemini", mode="single", providers=["gemini"]),
    BaselineConfig(name="single_kimi", mode="single", providers=["kimi"]),
    BaselineConfig(name="single_deepseek", mode="single", providers=["deepseek"]),
    BaselineConfig(name="single_grok", mode="single", providers=["grok"]),
    BaselineConfig(name="single_mistral", mode="single", providers=["mistral"]),
    BaselineConfig(name="routed_frontier", mode="routed", providers=["openai", "anthropic", "gemini"]),
    BaselineConfig(
        name="routed_frontier_plus",
        mode="routed",
        providers=["openai", "anthropic", "gemini", "deepseek", "grok", "mistral", "kimi"],
    ),
]

KEY_VARS = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "kimi": ["KIMI_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "grok": ["GROK_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def key_status() -> dict[str, bool]:
    return {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "KIMI_API_KEY": bool(os.getenv("KIMI_API_KEY")),
        "DEEPSEEK_API_KEY": bool(os.getenv("DEEPSEEK_API_KEY")),
        "GROK_API_KEY": bool(os.getenv("GROK_API_KEY")),
        "MISTRAL_API_KEY": bool(os.getenv("MISTRAL_API_KEY")),
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
    }


def config_required_vars(providers: list[str]) -> list[str]:
    vars_needed: list[str] = []
    for provider in providers:
        vars_needed.extend(KEY_VARS.get(provider.lower(), []))
    return sorted(set(vars_needed))


def config_has_keys(providers: list[str]) -> bool:
    for provider in providers:
        if provider.lower() not in KEY_VARS:
            continue
        if not any(os.getenv(var) for var in KEY_VARS[provider.lower()]):
            return False
    return True


def parse_configs(raw: str) -> list[BaselineConfig]:
    selected = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not selected:
        return DEFAULT_CONFIGS

    by_name = {cfg.name: cfg for cfg in DEFAULT_CONFIGS}
    configs: list[BaselineConfig] = []
    for name in selected:
        cfg = by_name.get(name)
        if cfg is None:
            raise ValueError(
                f"Unknown config '{name}'. Valid options: {', '.join(sorted(by_name.keys()))}"
            )
        configs.append(cfg)
    return configs


def capture_baselines(
    questions_path: str,
    output_dir: str,
    label: str,
    configs: list[BaselineConfig],
    max_questions: int | None,
    skip_missing_keys: bool,
    min_sources: int | None = None,
    max_latency_s: int | None = None,
    max_cost_usd: float = 3.0,
    parallel_workers: int = 1,
    parallel_configs: int = 1,
) -> dict[str, Any]:
    run_slug = f"{label}_{timestamp_slug()}"
    destination = Path(output_dir) / run_slug
    destination.mkdir(parents=True, exist_ok=True)

    if max(1, parallel_configs) > 1 and any(config.model_overrides for config in configs):
        raise ValueError("parallel-configs > 1 is not supported when model_overrides are used.")

    manifest: dict[str, Any] = {
        "label": label,
        "run_slug": run_slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "questions_path": questions_path,
        "max_questions": max_questions,
        "constraints": {
            "min_sources": min_sources,
            "max_latency_s": max_latency_s,
            "max_cost_usd": max_cost_usd,
        },
        "key_status": key_status(),
        "configs": [],
    }

    def _run_config(config: BaselineConfig) -> dict[str, Any]:
        config_record: dict[str, Any] = {
            "name": config.name,
            "mode": config.mode,
            "providers": config.providers,
            "model_overrides": config.model_overrides or {},
            "required_key_vars": config_required_vars(config.providers),
        }

        if skip_missing_keys and not config_has_keys(config.providers):
            config_record["status"] = "skipped_missing_keys"
            return config_record

        previous: dict[str, str | None] = {}
        for env_name, env_value in (config.model_overrides or {}).items():
            previous[env_name] = os.getenv(env_name)
            os.environ[env_name] = env_value
        try:
            result = run_benchmark_with_predictions(
                questions_path=questions_path,
                mode=config.mode,
                providers=config.providers,
                max_questions=max_questions,
                min_sources=min_sources,
                max_latency_s=max_latency_s,
                max_cost_usd=max_cost_usd,
                parallel_workers=parallel_workers,
            )
        finally:
            for env_name, old_value in previous.items():
                if old_value is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = old_value

        predictions_file = f"predictions_{config.name}.json"
        report_file = f"report_{config.name}.json"
        failures_file = f"failures_{config.name}.json"

        (destination / predictions_file).write_text(json.dumps(result["predictions"], indent=2))
        (destination / report_file).write_text(json.dumps(result["report"], indent=2))
        (destination / failures_file).write_text(json.dumps(result.get("failures", []), indent=2))

        config_record.update(
            {
                "status": "completed",
                "predictions_file": predictions_file,
                "report_file": report_file,
                "failures_file": failures_file,
                "num_predictions": len(result["predictions"]),
                "failure_count": len(result.get("failures", [])),
                "failed_question_ids": [item.get("id") for item in result.get("failures", []) if item.get("id")],
                "average_rubric_score": result["report"].get("average_rubric_score", 0.0),
                "brier_score": result["report"].get("calibration", {}).get("brier_score", 0.0),
                "ece": result["report"].get("calibration", {}).get("ece", 0.0),
                "avg_citation_coverage": result.get("quality_summary", {}).get("avg_citation_coverage", 0.0),
                "avg_unsupported_claims": result.get("quality_summary", {}).get("avg_unsupported_claims", 0.0),
                "avg_contradiction_flags": result.get("quality_summary", {}).get("avg_contradiction_flags", 0.0),
                "max_unsupported_claims": result.get("quality_summary", {}).get("max_unsupported_claims", 0),
                "max_contradiction_flags": result.get("quality_summary", {}).get("max_contradiction_flags", 0),
                "total_cost_usd": result.get("usage_summary", {}).get("total_cost_usd", 0.0),
                "avg_cost_usd": result.get("usage_summary", {}).get("avg_cost_usd", 0.0),
                "total_latency_s": result.get("usage_summary", {}).get("total_latency_s", 0.0),
                "avg_latency_s": result.get("usage_summary", {}).get("avg_latency_s", 0.0),
                "total_tokens_input": result.get("usage_summary", {}).get("total_tokens_input", 0),
                "total_tokens_output": result.get("usage_summary", {}).get("total_tokens_output", 0),
                "cost_estimated": result.get("usage_summary", {}).get("cost_estimated", True),
                "token_usage_partial": result.get("usage_summary", {}).get("token_usage_partial", False),
                "token_usage_notes": result.get("usage_summary", {}).get("token_usage_notes", ""),
            }
        )
        return config_record

    if max(1, parallel_configs) == 1:
        for config in configs:
            manifest["configs"].append(_run_config(config))
    else:
        by_name: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, parallel_configs)) as executor:
            futures = {executor.submit(_run_config, config): config.name for config in configs}
            for future in as_completed(futures):
                by_name[futures[future]] = future.result()
        for config in configs:
            manifest["configs"].append(by_name[config.name])

    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
