import socket
import logging
from typing import Dict, Any, Optional
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as rest_models
from app.core.config import settings

logger = logging.getLogger(__name__)

class QdrantManager:
    """Utility manager for Qdrant client connections with instant embedded fallback."""

    def __init__(self):
        self.host = settings.QDRANT_HOST
        self.port = settings.QDRANT_PORT
        self.api_key = settings.QDRANT_API_KEY
        self.collection_name = settings.QDRANT_COLLECTION
        self._cached_client: Optional[QdrantClient] = None

    def _is_server_reachable(self, timeout: float = 0.1) -> bool:
        """Fast TCP socket check to verify if Qdrant server is accepting connections."""
        target_ip = "127.0.0.1" if self.host == "localhost" else self.host
        try:
            with socket.create_connection((target_ip, self.port), timeout=timeout):
                return True
        except (socket.timeout, OSError):
            return False

    def get_client(self) -> QdrantClient:
        """Returns a Qdrant client. Connects to server if available; falls back to embedded in-memory Qdrant."""
        if self._cached_client is not None:
            return self._cached_client

        if self._is_server_reachable():
            try:
                client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key if self.api_key else None,
                    check_compatibility=False
                )
                self._cached_client = client
                logger.info(f"Connected to Qdrant server at {self.host}:{self.port}")
                return client
            except Exception as e:
                logger.warning(f"Error connecting to Qdrant server: {e}")

        logger.info("Qdrant server unreachable. Using embedded in-memory Qdrant instance.")
        self._cached_client = QdrantClient(location=":memory:")
        return self._cached_client

    def get_async_client(self) -> AsyncQdrantClient:
        """Returns an asynchronous Qdrant client instance."""
        return AsyncQdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key if self.api_key else None
        )

    def verify_connection(self) -> Dict[str, Any]:
        """Verifies connection to Qdrant cluster and returns health info."""
        try:
            client = self.get_client()
            collections_res = client.get_collections()
            collection_names = [col.name for col in collections_res.collections]
            return {
                "status": "connected",
                "host": self.host,
                "port": self.port,
                "collections_count": len(collection_names),
                "collections": collection_names
            }
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def ensure_collection_exists(
        self,
        collection_name: str = None,
        vector_size: int = 512,
        distance: rest_models.Distance = rest_models.Distance.COSINE
    ) -> bool:
        """Ensures that the specified Qdrant collection exists, creating it if necessary."""
        target_collection = collection_name or self.collection_name
        client = self.get_client()
        try:
            exists = client.collection_exists(target_collection)
            if not exists:
                logger.info(f"Creating Qdrant collection: {target_collection} (vector_size={vector_size})")
                client.create_collection(
                    collection_name=target_collection,
                    vectors_config=rest_models.VectorParams(
                        size=vector_size,
                        distance=distance
                    )
                )
            return True
        except Exception as e:
            logger.error(f"Error checking/creating collection '{target_collection}': {e}")
            return False

qdrant_manager = QdrantManager()
