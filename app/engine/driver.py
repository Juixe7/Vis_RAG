import httpx
from typing import AsyncGenerator, Optional
from app.core.config import settings

class LLMStreamDriver:
    def __init__(self, provider: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or settings.LLM_PROVIDER
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model = model or settings.LLM_MODEL

    async def generate_stream(self, prompt: str, image_base64: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Streams tokens from LLM backend (Ollama / vLLM)."""
        if self.provider == "ollama":
            async for token in self._stream_ollama(prompt, image_base64):
                yield token
        else:
            async for token in self._stream_vllm(prompt, image_base64):
                yield token

    async def _stream_ollama(self, prompt: str, image_base64: Optional[str] = None) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }
        if image_base64:
            payload["images"] = [image_base64]

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("POST", url, json=payload, timeout=60.0) as response:
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            data = json.loads(line)
                            yield data.get("response", "")
            except Exception as e:
                yield f"[Engine Stream Error: {str(e)}]"

    async def _stream_vllm(self, prompt: str, image_base64: Optional[str] = None) -> AsyncGenerator[str, None]:
        # vLLM OpenAI-compatible completion endpoint streaming placeholder
        yield f"vLLM streaming output for prompt: {prompt[:30]}..."
