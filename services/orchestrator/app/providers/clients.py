from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


class ProviderClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    provider: str
    model: str
    text: str
    success: bool
    tokens_input: int = 0
    tokens_input_cached: int = 0
    tokens_output: int = 0
    error: str | None = None


class BaseProviderClient:
    provider: str

    def generate(self, prompt: str, max_tokens: int = 800, timeout_s: float = 25.0) -> GenerationResult:
        raise NotImplementedError


class OpenAIClient(BaseProviderClient):
    provider = "openai"

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.2")
        if not self.api_key:
            raise ProviderClientError("OPENAI_API_KEY is not set")

    def generate(self, prompt: str, max_tokens: int = 800, timeout_s: float = 25.0) -> GenerationResult:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        # GPT-5-family chat completions expect max_completion_tokens.
        if self.model.startswith("gpt-5"):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {}) or {}
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text=text,
                success=True,
                tokens_input=int(usage.get("prompt_tokens", 0) or 0),
                tokens_input_cached=int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0),
                tokens_output=int(usage.get("completion_tokens", 0) or 0),
            )
        except Exception as exc:  # noqa: BLE001
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text="",
                success=False,
                error=str(exc),
            )


class AnthropicClient(BaseProviderClient):
    provider = "anthropic"

    def __init__(self) -> None:
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
        if not self.api_key:
            raise ProviderClientError("ANTHROPIC_API_KEY is not set")

    def generate(self, prompt: str, max_tokens: int = 800, timeout_s: float = 25.0) -> GenerationResult:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            data = response.json()
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            usage = data.get("usage", {}) or {}
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text=text,
                success=True,
                tokens_input=int(usage.get("input_tokens", 0) or 0),
                tokens_input_cached=int(usage.get("cache_read_input_tokens", 0) or 0),
                tokens_output=int(usage.get("output_tokens", 0) or 0),
            )
        except Exception as exc:  # noqa: BLE001
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text="",
                success=False,
                error=str(exc),
            )


class KimiClient(BaseProviderClient):
    provider = "kimi"

    def __init__(self) -> None:
        self.api_key = os.getenv("KIMI_API_KEY")
        self.model = os.getenv("KIMI_MODEL", "kimi-k2.5")
        self.base_url = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai").rstrip("/")
        # Kimi K2 variants require temperature=1 on this endpoint.
        self.temperature = float(os.getenv("KIMI_TEMPERATURE", "1.0"))
        if not self.api_key:
            raise ProviderClientError("KIMI_API_KEY is not set")

    def generate(self, prompt: str, max_tokens: int = 800, timeout_s: float = 25.0) -> GenerationResult:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {}) or {}
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text=text,
                success=True,
                tokens_input=int(usage.get("prompt_tokens", 0) or 0),
                tokens_input_cached=int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0),
                tokens_output=int(usage.get("completion_tokens", 0) or 0),
            )
        except Exception as exc:  # noqa: BLE001
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text="",
                success=False,
                error=str(exc),
            )


class GeminiClient(BaseProviderClient):
    provider = "gemini"

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
        if not self.api_key:
            raise ProviderClientError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")

    def generate(self, prompt: str, max_tokens: int = 800, timeout_s: float = 25.0) -> GenerationResult:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
        }
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts)
            usage = data.get("usageMetadata", {}) or {}
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text=text,
                success=True,
                tokens_input=int(usage.get("promptTokenCount", 0) or 0),
                tokens_output=int(usage.get("candidatesTokenCount", 0) or 0),
            )
        except Exception as exc:  # noqa: BLE001
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text="",
                success=False,
                error=str(exc),
            )


def make_provider_client(provider: str) -> BaseProviderClient:
    normalized = provider.lower()
    if normalized == "openai":
        return OpenAIClient()
    if normalized == "anthropic":
        return AnthropicClient()
    if normalized == "kimi":
        return KimiClient()
    if normalized in {"gemini", "google"}:
        return GeminiClient()
    raise ProviderClientError(f"Unsupported provider: {provider}")


def generate_text(provider: str, prompt: str, max_tokens: int = 800) -> GenerationResult:
    try:
        client = make_provider_client(provider)
    except Exception as exc:  # noqa: BLE001
        return GenerationResult(
            provider=provider,
            model="unknown",
            text="",
            success=False,
            error=str(exc),
        )
    return client.generate(prompt=prompt, max_tokens=max_tokens)
