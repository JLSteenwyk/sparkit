from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    configured: bool
    env_var: str


class ProviderRegistry:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping

    def resolve(self, providers: list[str]) -> list[ProviderStatus]:
        statuses: list[ProviderStatus] = []
        for provider in providers:
            key_candidates = self.mapping.get(provider.lower())
            if not key_candidates:
                statuses.append(ProviderStatus(provider=provider, configured=False, env_var="unsupported"))
                continue

            configured_env = ""
            configured = False
            for env_name in key_candidates:
                if os.getenv(env_name):
                    configured = True
                    configured_env = env_name
                    break

            statuses.append(
                ProviderStatus(
                    provider=provider,
                    configured=configured,
                    env_var=configured_env if configured else key_candidates[0],
                )
            )
        return statuses


def build_default_registry() -> ProviderRegistry:
    return ProviderRegistry(
        {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "kimi": ["KIMI_API_KEY"],
            "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        }
    )
