from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import requests

from sparkit.evidence.schema import EvidenceItem, EvidenceType, StudyType
from sparkit.providers.base import ProviderQuery


class ExaProvider:
    name = "exa"

    def search(self, query: ProviderQuery) -> list[EvidenceItem]:
        api_key = os.getenv("EXA_API_KEY", "").strip()
        if not api_key:
            return []

        timeout_s = float(os.getenv("EXA_TIMEOUT_S", "25"))
        max_results = max(1, min(25, query.max_items))
        url = os.getenv("EXA_SEARCH_URL", "https://api.exa.ai/search")
        payload = {
            "query": self._science_query(query.question),
            "numResults": max_results,
            "type": "auto",
            "text": True,
        }
        headers = {"x-api-key": api_key, "content-type": "application/json"}

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
        except Exception:  # noqa: BLE001
            return []

        raw_results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(raw_results, list):
            return []

        items: list[EvidenceItem] = []
        for row in raw_results[:max_results]:
            if not isinstance(row, dict):
                continue
            title = _s(row.get("title"))
            claim = _s(row.get("text")) or _s(row.get("snippet")) or title
            if not claim:
                continue
            url_value = _s(row.get("url"))
            year = _year_from_any(row.get("publishedDate"))
            items.append(
                EvidenceItem(
                    provider=self.name,
                    claim=claim,
                    evidence_type=EvidenceType.NEUTRAL,
                    study_type=_infer_study_type(title=title, claim=claim),
                    confidence=0.55,
                    title=title or None,
                    url=url_value or None,
                    year=year,
                    provenance={"source": "exa_search", "raw_score": row.get("score")},
                )
            )
        return items

    @staticmethod
    def _science_query(question: str) -> str:
        q = " ".join(question.split())
        hint = os.getenv("EXA_SCIENCE_HINT", "site:pubmed.ncbi.nlm.nih.gov OR doi OR clinical trial OR systematic review")
        return f"{q} {hint}".strip()


def _s(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _year_from_any(value: Any) -> int | None:
    text = _s(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text, fmt).year
        except ValueError:
            continue
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def _infer_study_type(*, title: str, claim: str) -> StudyType:
    blob = f"{title} {claim}".lower()
    if "meta-analysis" in blob or "meta analysis" in blob:
        return StudyType.META_ANALYSIS
    if "systematic review" in blob:
        return StudyType.SYSTEMATIC_REVIEW
    if "randomized" in blob or "randomised" in blob:
        return StudyType.RANDOMIZED_TRIAL
    if "cohort" in blob or "observational" in blob:
        return StudyType.OBSERVATIONAL
    if "mouse" in blob or "in vitro" in blob or "cell line" in blob:
        return StudyType.PRECLINICAL
    if "case report" in blob:
        return StudyType.CASE_REPORT
    if "preprint" in blob or "arxiv" in blob or "biorxiv" in blob:
        return StudyType.PREPRINT
    return StudyType.UNKNOWN

