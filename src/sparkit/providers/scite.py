from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any

import requests

from sparkit.evidence.schema import EvidenceItem, EvidenceType, StudyType
from sparkit.providers.base import ProviderQuery


class SciteProvider:
    name = "scite"

    def search(self, query: ProviderQuery) -> list[EvidenceItem]:
        api_key = os.getenv("EXA_API_KEY", "").strip()
        if not api_key:
            return []
        timeout_s = float(os.getenv("SCITE_TIMEOUT_S", "20"))
        max_results = max(1, min(20, query.max_items))

        # Discovery constrained to scite reports.
        search_url = os.getenv("EXA_SEARCH_URL", "https://api.exa.ai/search")
        payload = {
            "query": f"{' '.join(query.question.split())} site:scite.ai/reports",
            "numResults": max_results,
            "type": "auto",
        }
        headers = {"x-api-key": api_key, "content-type": "application/json"}
        try:
            resp = requests.post(search_url, json=payload, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
        except Exception:  # noqa: BLE001
            return []

        rows = data.get("results") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return []

        items: list[EvidenceItem] = []
        for row in rows[:max_results]:
            if not isinstance(row, dict):
                continue
            url = _s(row.get("url"))
            if "scite.ai/reports/" not in url:
                continue
            title = _s(row.get("title"))
            text = _s(row.get("text")) or _s(row.get("snippet"))
            support_n, contrast_n = _fetch_scite_tallies(url=url, timeout_s=timeout_s)
            ev_type = EvidenceType.NEUTRAL
            if contrast_n > support_n and contrast_n >= 1:
                ev_type = EvidenceType.CONTRADICTING
            elif support_n >= contrast_n and support_n >= 1:
                ev_type = EvidenceType.SUPPORTING
            conf = min(0.9, 0.45 + (0.03 * min(10, support_n + contrast_n)))
            year = _year_from_any(row.get("publishedDate"))
            items.append(
                EvidenceItem(
                    provider=self.name,
                    claim=text or title or url,
                    evidence_type=ev_type,
                    study_type=_infer_study_type(title=title, claim=text),
                    confidence=conf,
                    title=title or None,
                    url=url or None,
                    year=year,
                    provenance={
                        "source": "scite_report",
                        "supporting_citations": support_n,
                        "contrasting_citations": contrast_n,
                    },
                )
            )
        return items


def _fetch_scite_tallies(*, url: str, timeout_s: float) -> tuple[int, int]:
    try:
        r = requests.get(url, timeout=timeout_s)
        r.raise_for_status()
        html = r.text
    except Exception:  # noqa: BLE001
        return 0, 0

    support = _extract_count(html, "Supporting")
    contrast = _extract_count(html, "Contrasting")
    return support, contrast


def _extract_count(html: str, label: str) -> int:
    patterns = [
        rf"{label}\s*</[^>]+>\s*([0-9]+)",
        rf"{label}\s*([0-9]+)",
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return 0


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
