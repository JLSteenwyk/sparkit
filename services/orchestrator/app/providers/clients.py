from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter

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
    latency_s: float = 0.0
    error: str | None = None


class BaseProviderClient:
    provider: str

    def generate(self, prompt: str, max_tokens: int = 800, timeout_s: float = 25.0) -> GenerationResult:
        raise NotImplementedError


def _default_timeout_for_provider(provider: str) -> float:
    tuned_defaults = {
        "grok": 60.0,
        "deepseek": 45.0,
    }
    per_provider = os.getenv(f"{provider.upper()}_TIMEOUT_S")
    if per_provider:
        try:
            return max(1.0, float(per_provider))
        except ValueError:
            pass
    default_timeout = tuned_defaults.get(provider.lower(), 35.0)
    global_default = os.getenv("SPARKIT_PROVIDER_TIMEOUT_S", str(default_timeout))
    try:
        return max(1.0, float(global_default))
    except ValueError:
        return default_timeout


class OpenAICompatibleClient(BaseProviderClient):
    api_key: str
    model: str
    base_url: str
    temperature: float
    api_key_env_name: str
    default_base_url: str
    default_model: str
    provider: str

    def __init__(self) -> None:
        self.api_key = os.getenv(self.api_key_env_name)
        self.model = os.getenv(f"{self.provider.upper()}_MODEL", self.default_model)
        self.base_url = os.getenv(f"{self.provider.upper()}_BASE_URL", self.default_base_url).rstrip("/")
        self.temperature = float(os.getenv(f"{self.provider.upper()}_TEMPERATURE", "0.2"))
        if not self.api_key:
            raise ProviderClientError(f"{self.api_key_env_name} is not set")

    def generate(self, prompt: str, max_tokens: int = 800, timeout_s: float = 25.0) -> GenerationResult:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        started = perf_counter()
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            data = response.json()
            message = (data.get("choices") or [{}])[0].get("message") or {}
            text = str(message.get("content", "") or "")
            if not text.strip():
                reasoning = str(message.get("reasoning_content", "") or "")
                if self.provider == "deepseek" and reasoning.strip():
                    text = reasoning
                else:
                    reason_suffix = " with reasoning_content" if reasoning.strip() else ""
                    return GenerationResult(
                        provider=self.provider,
                        model=self.model,
                        text="",
                        success=False,
                        latency_s=max(0.0, perf_counter() - started),
                        error=f"empty_content{reason_suffix}",
                    )
            usage = data.get("usage", {}) or {}
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text=text,
                success=True,
                tokens_input=int(usage.get("prompt_tokens", 0) or 0),
                tokens_input_cached=int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0),
                tokens_output=int(usage.get("completion_tokens", 0) or 0),
                latency_s=max(0.0, perf_counter() - started),
            )
        except Exception as exc:  # noqa: BLE001
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text="",
                success=False,
                latency_s=max(0.0, perf_counter() - started),
                error=str(exc),
            )


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
        started = perf_counter()
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            data = response.json()
            message = (data.get("choices") or [{}])[0].get("message") or {}
            text = str(message.get("content", "") or "")
            if not text.strip():
                reasoning = str(message.get("reasoning_content", "") or "")
                if reasoning.strip():
                    text = reasoning
                else:
                    return GenerationResult(
                        provider=self.provider,
                        model=self.model,
                        text="",
                        success=False,
                        error="empty_content",
                    )
            usage = data.get("usage", {}) or {}
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text=text,
                success=True,
                tokens_input=int(usage.get("prompt_tokens", 0) or 0),
                tokens_input_cached=int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0),
                tokens_output=int(usage.get("completion_tokens", 0) or 0),
                latency_s=max(0.0, perf_counter() - started),
            )
        except Exception as exc:  # noqa: BLE001
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text="",
                success=False,
                latency_s=max(0.0, perf_counter() - started),
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
        started = perf_counter()
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            data = response.json()
            if str(data.get("stop_reason", "")).lower() == "refusal":
                return GenerationResult(
                    provider=self.provider,
                    model=self.model,
                    text="",
                    success=False,
                    latency_s=max(0.0, perf_counter() - started),
                    error="refusal",
                )
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
                latency_s=max(0.0, perf_counter() - started),
            )
        except Exception as exc:  # noqa: BLE001
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text="",
                success=False,
                latency_s=max(0.0, perf_counter() - started),
                error=str(exc),
            )


class KimiClient(BaseProviderClient):
    provider = "kimi"

    def __init__(self) -> None:
        self.api_key = os.getenv("KIMI_API_KEY")
        self.model = os.getenv("KIMI_MODEL", "kimi-k2-turbo-preview")
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
        started = perf_counter()
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            data = response.json()
            message = (data.get("choices") or [{}])[0].get("message") or {}
            text = str(message.get("content", "") or "")
            if not text.strip():
                reasoning = str(message.get("reasoning_content", "") or "")
                if reasoning.strip():
                    text = reasoning
                else:
                    return GenerationResult(
                        provider=self.provider,
                        model=self.model,
                        text="",
                        success=False,
                        latency_s=max(0.0, perf_counter() - started),
                        error="empty_content",
                    )
            usage = data.get("usage", {}) or {}
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text=text,
                success=True,
                tokens_input=int(usage.get("prompt_tokens", 0) or 0),
                tokens_input_cached=int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0),
                tokens_output=int(usage.get("completion_tokens", 0) or 0),
                latency_s=max(0.0, perf_counter() - started),
            )
        except Exception as exc:  # noqa: BLE001
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text="",
                success=False,
                latency_s=max(0.0, perf_counter() - started),
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
        started = perf_counter()
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
                latency_s=max(0.0, perf_counter() - started),
            )
        except Exception as exc:  # noqa: BLE001
            return GenerationResult(
                provider=self.provider,
                model=self.model,
                text="",
                success=False,
                latency_s=max(0.0, perf_counter() - started),
                error=str(exc),
            )


class DeepSeekClient(OpenAICompatibleClient):
    provider = "deepseek"
    api_key_env_name = "DEEPSEEK_API_KEY"
    default_base_url = "https://api.deepseek.com"
    default_model = "deepseek-reasoner"


class GrokClient(OpenAICompatibleClient):
    provider = "grok"
    api_key_env_name = "GROK_API_KEY"
    default_base_url = "https://api.x.ai"
    default_model = "grok-4-0709"


class MistralClient(OpenAICompatibleClient):
    provider = "mistral"
    api_key_env_name = "MISTRAL_API_KEY"
    default_base_url = "https://api.mistral.ai"
    default_model = "mistral-large-2512"


def make_provider_client(provider: str) -> BaseProviderClient:
    normalized = provider.lower()
    if normalized == "openai":
        return OpenAIClient()
    if normalized == "anthropic":
        return AnthropicClient()
    if normalized == "kimi":
        return KimiClient()
    if normalized == "deepseek":
        return DeepSeekClient()
    if normalized == "grok":
        return GrokClient()
    if normalized == "mistral":
        return MistralClient()
    if normalized in {"gemini", "google"}:
        return GeminiClient()
    raise ProviderClientError(f"Unsupported provider: {provider}")


def generate_text(provider: str, prompt: str, max_tokens: int = 800, timeout_s: float | None = None) -> GenerationResult:
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
    resolved_timeout = timeout_s if timeout_s is not None else _default_timeout_for_provider(provider)
    return client.generate(prompt=prompt, max_tokens=max_tokens, timeout_s=resolved_timeout)
