import os
import io
import json
import base64
import logging
import functools
from typing import List, Union, Optional
from app.core.config import settings
import numpy as np

logger = logging.getLogger(__name__)

# Redis cache client optional initialization
try:
    import redis
    from app.core.config import settings
    _redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=0.5)
    _redis_client.ping()
    logger.info("Connected to Redis for query vector caching.")
except Exception:
    _redis_client = None

class CLIPIndexer:
    """CLIP Vector Indexer with LRU & Redis caching for producing normalized text and image embeddings using clip-ViT-B-32."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or getattr(settings, "EMBEDDING_MODEL", "sentence-transformers/clip-ViT-B-32")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading CLIP model: {self.model_name}...")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load CLIP model '{self.model_name}': {e}")
                raise e
        return self._model

    def _normalize(self, vector: np.ndarray) -> List[float]:
        """Applies L2 normalization to vector embedding."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.tolist()
        return (vector / norm).tolist()

    @functools.lru_cache(maxsize=1024)
    def _cached_text_embedding(self, text: str) -> List[float]:
        """Internal cached text embedding generator."""
        embedding = self.model.encode(text)
        return self._normalize(np.array(embedding, dtype=np.float32))

    def get_text_embedding(self, text: str) -> List[float]:
        """Generates normalized 512-dimensional vector embedding for text with sub-5ms LRU & Redis caching."""
        if not text or not text.strip():
            return [0.0] * 512

        clean_text = text.strip()

        # 1. Redis cache lookup
        if _redis_client:
            try:
                cache_key = f"clip_vec:{hash(clean_text)}"
                cached_val = _redis_client.get(cache_key)
                if cached_val:
                    return json.loads(cached_val)
            except Exception:
                pass

        # 2. In-memory LRU cache lookup
        vec = self._cached_text_embedding(clean_text)

        # Write back to Redis if available
        if _redis_client:
            try:
                cache_key = f"clip_vec:{hash(clean_text)}"
                _redis_client.setex(cache_key, 86400, json.dumps(vec))
            except Exception:
                pass

        return vec

    def get_image_embedding(self, image_input: Union[str, bytes]) -> List[float]:
        """Generates normalized 512-dimensional vector embedding for an image (path, bytes, or base64)."""
        from PIL import Image

        image = None
        if isinstance(image_input, str):
            if os.path.exists(image_input):
                image = Image.open(image_input).convert("RGB")
            elif len(image_input) > 100:
                # Handle base64 string
                b64_str = image_input.split(",", 1)[1] if "," in image_input else image_input
                image_bytes = base64.b64decode(b64_str)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            else:
                raise FileNotFoundError(f"Image path not found: {image_input}")
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        embedding = self.model.encode(image)
        return self._normalize(np.array(embedding, dtype=np.float32))

clip_indexer = CLIPIndexer()
