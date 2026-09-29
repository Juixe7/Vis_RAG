import logging
import asyncio
from typing import List, Optional, Dict, Any, Union
from app.models.schemas import SearchResult, HybridSearchQuery, DocumentChunk
from app.core.config import settings
from app.core.qdrant_client import qdrant_manager
from app.retrieval.clip_indexer import clip_indexer
from app.retrieval.bm25_search import bm25_indexer

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Hybrid Retrieval Engine combining Qdrant dense vector search, BM25 sparse search, and Reciprocal Rank Fusion (RRF)."""

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 10
    ) -> List[SearchResult]:
        """Queries Qdrant vector database using dense vector embedding."""
        client = qdrant_manager.get_client()
        try:
            if not client.collection_exists(self.collection_name):
                logger.warning(f"Collection '{self.collection_name}' does not exist in Qdrant.")
                return []

            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k
                )
                search_res = response.points
            elif hasattr(client, "search"):
                search_res = client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k
                )
            else:
                logger.error("QdrantClient has neither query_points nor search method.")
                return []

            results = []
            for point in search_res:
                payload = point.payload or {}
                results.append(
                    SearchResult(
                        doc_id=payload.get("doc_id") or str(point.id),
                        score=float(point.score),
                        content=payload.get("content", ""),
                        image_path=payload.get("image_path"),
                        metadata=payload.get("metadata", {})
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Error during dense Qdrant search: {e}")
            return []

    def search_sparse(
        self,
        query_text: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Queries BM25 sparse index using tokenized query text."""
        return bm25_indexer.search(query=query_text, top_k=top_k)

    def reciprocal_rank_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        k: int = 60,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[SearchResult]:
        """Combines ranked results using weighted Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, SearchResult] = {}

        dense_weight = alpha * 2.0
        sparse_weight = (1.0 - alpha) * 2.0

        # Process dense rankings
        for rank, doc in enumerate(dense_results):
            doc_id = doc.doc_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + dense_weight * (1.0 / (k + rank + 1))
            doc_map[doc_id] = doc

        # Process sparse rankings
        for rank, doc in enumerate(sparse_results):
            doc_id = doc.doc_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + sparse_weight * (1.0 / (k + rank + 1))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # Sort documents by accumulated weighted RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        final_results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            original = doc_map[doc_id]
            final_results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=round(rrf_score, 6),
                    content=original.content,
                    image_path=original.image_path,
                    metadata=original.metadata
                )
            )

        return final_results

    def hybrid_search(
        self,
        query: Union[str, HybridSearchQuery],
        image_input: Optional[Union[str, bytes]] = None,
        top_k: int = 5,
        alpha: float = 0.5,
        rrf_k: int = 60
    ) -> List[SearchResult]:
        """Performs full hybrid retrieval pipeline for text or image query."""
        if isinstance(query, HybridSearchQuery):
            query_text = query.query
            image_input = image_input or query.image_base64
            top_k = query.top_k
            alpha = query.alpha
            rrf_k = query.rrf_k
        else:
            query_text = str(query)

        # 1. Compute query dense embedding (Text or Image)
        if image_input:
            query_vector = clip_indexer.get_image_embedding(image_input)
        else:
            query_vector = clip_indexer.get_text_embedding(query_text)

        # 2. Dense retrieval from Qdrant
        dense_results = self.search_dense(query_vector=query_vector, top_k=top_k * 2) if alpha > 0.0 else []

        # 3. Sparse retrieval from BM25
        sparse_results = self.search_sparse(query_text=query_text, top_k=top_k * 2) if (query_text and alpha < 1.0) else []

        # 4. Combine via weighted Reciprocal Rank Fusion (RRF)
        return self.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=rrf_k,
            top_k=top_k,
            alpha=alpha
        )

    async def hybrid_search_async(
        self,
        query: Union[str, HybridSearchQuery],
        image_input: Optional[Union[str, bytes]] = None,
        top_k: int = 5,
        alpha: float = 0.5,
        rrf_k: int = 60
    ) -> List[SearchResult]:
        """Async wrapper running CPU-bound hybrid search in a threadpool so main asyncio loop stays responsive."""
        return await asyncio.to_thread(self.hybrid_search, query, image_input, top_k, alpha, rrf_k)

hybrid_retriever = HybridRetriever()
