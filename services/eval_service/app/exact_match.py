from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from services.orchestrator.app.providers import generate_text


_WS_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[\s\.,;:]+|[\s\.,;:]+$")
_LIST_TOKEN_RE = re.compile(r"\d+")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clean(text: str) -> str:
    lowered = text.strip().lower()
    lowered = lowered.replace("’", "'")
    lowered = lowered.replace("–", "-")
    lowered = lowered.replace("—", "-")
    lowered = _WS_RE.sub(" ", lowered)
    return lowered


def _canonical_name(text: str) -> str:
    value = _clean(text)
    value = value.replace("racemic mixture", "racemate")
    value = value.replace("racemic", "racemate")
    value = value.replace("mixture of enantiomers", "racemate")
    value = value.replace(" of ", " ")
    value = value.replace("the ", "")
    value = _PUNCT_EDGE_RE.sub("", value)
    return value


def _canonical_numeric_lists(text: str) -> str | None:
    # Supports formats like "(1,4,5), (1,3,4,5,6)" or "1,4,5;1,3,4,5,6"
    groups = []
    paren_groups = re.findall(r"\(([^()]*)\)", text)
    if paren_groups:
        for chunk in paren_groups:
            nums = [int(tok) for tok in _LIST_TOKEN_RE.findall(chunk)]
            if nums:
                groups.append(tuple(nums))
    else:
        # fallback: one flat numeric sequence
        nums = [int(tok) for tok in _LIST_TOKEN_RE.findall(text)]
        if nums:
            groups = [tuple(nums)]
    if not groups:
        return None
    return "|".join(",".join(str(n) for n in grp) for grp in groups)


def _deterministic_match(gold: str, pred: str) -> bool:
    gold_name = _canonical_name(gold)
    pred_name = _canonical_name(pred)
    if not gold_name or not pred_name:
        return False
    if gold_name == pred_name:
        return True
    # containment for short canonical labels (e.g. "collagen", "no change")
    if len(gold_name.split()) <= 4 and gold_name in pred_name:
        return True

    gold_list = _canonical_numeric_lists(gold)
    pred_list = _canonical_numeric_lists(pred)
    if gold_list and pred_list and gold_list == pred_list:
        return True
    return False


def _llm_enabled() -> bool:
    return os.getenv("EXACTMATCH_ENABLE_LLM_GRADER", "1").strip().lower() not in {"0", "false", "no"}


@lru_cache(maxsize=8192)
def _llm_exact_match_anthropic(question: str, gold: str, pred: str) -> bool | None:
    if not _llm_enabled():
        return False
    prompt = (
        "You are grading an exact-match QA response.\n"
        "Return JSON only: {\"equivalent\": true|false}.\n"
        "Mark true only if the prediction clearly expresses the same final answer as gold.\n"
        "Ignore verbosity, casing, punctuation, and minor wording differences.\n"
        "Do not require rationale if final answer is present.\n\n"
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Prediction:\n{pred}\n"
    )
    model = os.getenv("EXACTMATCH_GRADER_MODEL_ANTHROPIC", "claude-sonnet-4-5")
    old = os.getenv("ANTHROPIC_MODEL")
    try:
        os.environ["ANTHROPIC_MODEL"] = model
        result = generate_text("anthropic", prompt, max_tokens=80)
    finally:
        if old is None:
            os.environ.pop("ANTHROPIC_MODEL", None)
        else:
            os.environ["ANTHROPIC_MODEL"] = old
    if not result.success or not result.text.strip():
        return None
    text = result.text.strip()
    match = _JSON_RE.search(text)
    if match:
        text = match.group(0)
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return bool(payload.get("equivalent", False))


@lru_cache(maxsize=8192)
def _llm_exact_match_openai(question: str, gold: str, pred: str) -> bool | None:
    if not _llm_enabled():
        return None
    prompt = (
        "You are grading an exact-match QA response.\n"
        "Return JSON only: {\"equivalent\": true|false}.\n"
        "Mark true only if the prediction clearly expresses the same final answer as gold.\n"
        "Ignore verbosity, casing, punctuation, and minor wording differences.\n"
        "Do not require rationale if final answer is present.\n\n"
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Prediction:\n{pred}\n"
    )
    model = os.getenv("EXACTMATCH_GRADER_MODEL_OPENAI", "gpt-5-mini")
    old = os.getenv("OPENAI_MODEL")
    try:
        os.environ["OPENAI_MODEL"] = model
        result = generate_text("openai", prompt, max_tokens=80)
    finally:
        if old is None:
            os.environ.pop("OPENAI_MODEL", None)
        else:
            os.environ["OPENAI_MODEL"] = old
    if not result.success or not result.text.strip():
        return None
    text = result.text.strip()
    match = _JSON_RE.search(text)
    if match:
        text = match.group(0)
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return bool(payload.get("equivalent", False))


def exact_match_correct(question: str, gold: str, pred: str) -> bool:
    anth = _llm_exact_match_anthropic(question, gold, pred)
    opai = _llm_exact_match_openai(question, gold, pred)
    if anth is None or opai is None:
        return False
    return anth is True and opai is True
