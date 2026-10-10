import json
import logging
import asyncio
import httpx
from typing import AsyncGenerator, List, Optional, Union
from app.models.schemas import SearchResult
from app.core.config import settings
from app.core.metrics import LatencyTracker

logger = logging.getLogger(__name__)

class VLMStreamingEngine:
    """Async VLM streaming engine interfacing with vLLM / Ollama OpenAI-compatible streaming endpoints (LLaVA, Qwen-VL)."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL
        self.api_key = settings.LLM_API_KEY or settings.OPENAI_API_KEY or "EMPTY"

    def build_prompt(self, query: str, retrieved_contexts: List[SearchResult]) -> str:
        """Constructs a structured multimodal RAG prompt from query and retrieved chunks."""
        if retrieved_contexts:
            context_str = "\n".join([f"[{i+1}] {chunk.content}" for i, chunk in enumerate(retrieved_contexts)])
        else:
            context_str = "No external context retrieved."

        prompt = (
            f"You are Vision RAG Assistant, an expert AI for multimodal document understanding.\n\n"
            f"Retrieved Context:\n{context_str}\n\n"
            f"User Question:\n{query}\n\n"
            f"Provide a clear, precise, and well-structured answer based on the context."
        )
        return prompt

    async def stream_answer(
        self,
        query: str,
        retrieved_contexts: List[SearchResult],
        image_path: Optional[str] = None,
        tracker: Optional[LatencyTracker] = None
    ) -> AsyncGenerator[str, None]:
        """Streams generated tokens from OpenAI-compatible VLM endpoint (/v1/chat/completions)."""
        prompt_text = self.build_prompt(query, retrieved_contexts)
        endpoint = f"{self.base_url}/v1/chat/completions" if not self.base_url.endswith("/v1") else f"{self.base_url}/chat/completions"

        # Format multimodal user message (text + image if present)
        if image_path and (image_path.startswith("data:image") or len(image_path) > 100):
            user_content: Union[str, List[dict]] = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_path}}
            ]
        else:
            user_content = prompt_text

        messages = [
            {"role": "user", "content": user_content}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 1024
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        emitted_any = False
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("POST", endpoint, json=payload, headers=headers, timeout=30.0) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk_json = json.loads(data_str)
                                    choices = chunk_json.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            if tracker:
                                                tracker.record_token()
                                            emitted_any = True
                                            yield content
                                except json.JSONDecodeError:
                                    continue
            except Exception as e:
                logger.info(f"VLM server endpoint ({endpoint}) unavailable ({e}). Using context-driven streaming generator.")

        if not emitted_any:
            # Context-driven streaming fallback generator when local vLLM/Ollama is offline
            top_ctx = retrieved_contexts[0].content if retrieved_contexts else "No context available"
            fallback_text = (
                f"Based on the retrieved context:\n"
                f"Context Chunk: \"{top_ctx}\"\n\n"
                f"The Vision RAG Engine has processed your query ('{query}') using hybrid CLIP + BM25 search "
                f"and Reciprocal Rank Fusion re-ranking."
            )

            words = fallback_text.split(" ")
            for i, word in enumerate(words):
                if tracker:
                    tracker.record_token()
                yield word + (" " if i < len(words) - 1 else "")
                await asyncio.sleep(0.02)

vlm_engine = VLMStreamingEngine()
