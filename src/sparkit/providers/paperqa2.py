from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

try:
    from paperqa import Docs  # type: ignore
except Exception:  # noqa: BLE001
    Docs = None  # type: ignore

from sparkit.evidence.schema import EvidenceItem, EvidenceType, StudyType
from sparkit.providers.base import ProviderQuery


class PaperQA2Provider:
    name = "paperqa2"
    _docs_cache: dict[str, Any] = {}

    def search(self, query: ProviderQuery) -> list[EvidenceItem]:
        if Docs is None:
            return []

        paper_dir = os.getenv("PAPERQA_PAPER_DIRECTORY", "").strip()
        if not paper_dir:
            return []

        docs = self._get_docs(paper_dir)
        if docs is None:
            return []

        try:
            response = docs.query(
                query.question,
                k=max(4, min(20, query.max_items)),
                max_sources=max(2, min(8, query.max_items)),
            )
        except Exception:  # noqa: BLE001
            return []

        contexts = self._extract_context_blocks(response)
        if not contexts:
            answer_text = self._extract_answer_text(response)
            if not answer_text:
                return []
            return [
                EvidenceItem(
                    provider=self.name,
                    claim=answer_text,
                    evidence_type=EvidenceType.NEUTRAL,
                    study_type=StudyType.UNKNOWN,
                    confidence=0.55,
                    provenance={"source": "paperqa2_answer_only"},
                )
            ]

        items: list[EvidenceItem] = []
        for block in contexts[: query.max_items]:
            title = None
            claim = None
            if isinstance(block, dict):
                title = str(block.get("title") or block.get("citation") or "") or None
                claim = str(block.get("text") or block.get("summary") or block.get("context") or "") or None
            elif isinstance(block, str):
                claim = block
            if not claim:
                continue
            items.append(
                EvidenceItem(
                    provider=self.name,
                    claim=claim,
                    evidence_type=EvidenceType.NEUTRAL,
                    study_type=StudyType.UNKNOWN,
                    confidence=0.65,
                    title=title,
                    provenance={"source": "paperqa2_context"},
                )
            )
        return items

    def _get_docs(self, paper_dir: str) -> Any | None:
        llm_model = os.getenv("PAPERQA_LLM_MODEL", "").strip() or "gpt-4-0125-preview"
        if llm_model.startswith("gpt-5"):
            # paper-qa 4.9.0 uses a completions-style path for this adapter flow.
            # gpt-5 chat models are incompatible there, so force a known-good fallback.
            fallback = "gpt-4-0125-preview"
            print(
                f"[paperqa2] PAPERQA_LLM_MODEL={llm_model} is incompatible with paper-qa adapter path; "
                f"falling back to {fallback}",
                file=sys.stderr,
            )
            llm_model = fallback
        cache_key = f"{paper_dir}::{llm_model}"
        cached = self._docs_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            docs = Docs(llm=llm_model)
            root = Path(paper_dir)
            if not root.exists() or not root.is_dir():
                return None
            paths = sorted(
                [
                    *root.glob("*.pdf"),
                    *root.glob("*.txt"),
                    *root.glob("*.md"),
                    *root.glob("*.html"),
                ]
            )
            for path in paths:
                try:
                    docs.add(path)
                except Exception:  # noqa: BLE001
                    continue
            self._docs_cache[cache_key] = docs
            return docs
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _extract_answer_text(response: Any) -> str:
        for field in ("answer", "formatted_answer"):
            value = getattr(response, field, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if isinstance(response, str):
            return response.strip()
        return ""

    @staticmethod
    def _extract_context_blocks(response: Any) -> list[Any]:
        context = getattr(response, "context", None)
        if context is None:
            return []
        if isinstance(context, list):
            return context
        if isinstance(context, tuple):
            return list(context)
        if isinstance(context, str) and context.strip():
            return [context.strip()]
        return []
