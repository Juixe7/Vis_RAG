from typing import List, Dict, Any, Optional
from app.models.schemas import SearchResult
from app.core.config import settings

class HybridRetriever:
    """Hybrid Retriever combining Qdrant dense vector search, BM25 sparse keyword search, and RRF re-ranking."""

    def __init__(self):
        self.qdrant_host = settings.QDRANT_HOST
        self.qdrant_port = settings.QDRANT_PORT
        self.collection_name = settings.QDRANT_COLLECTION

    def reciprocal_rank_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        k: int = 60,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Combines dense and sparse ranked lists using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, SearchResult] = {}

        for rank, doc in enumerate(dense_results):
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc.doc_id] = doc

        for rank, doc in enumerate(sparse_results):
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc.doc_id] = doc

        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        final_results = []
        for doc_id, score in sorted_docs[:top_k]:
            original_doc = doc_map[doc_id]
            final_results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=round(score, 4),
                    content=original_doc.content,
                    metadata=original_doc.metadata
                )
            )

        return final_results

    async def search(self, query: str, image_base64: Optional[str] = None, top_k: int = 5) -> List[SearchResult]:
        """Performs hybrid retrieval."""
        # Demo / Placeholder retrieval results until Qdrant collection is connected
        dummy_dense = [
            SearchResult(doc_id="doc_1", score=0.92, content="Vision-Language models integrate image embeddings with text.", metadata={"source": "paper.pdf"}),
            SearchResult(doc_id="doc_2", score=0.85, content="Qdrant supports high-performance vector indexing with HNSW.", metadata={"source": "qdrant_docs.md"}),
        ]
        dummy_sparse = [
            SearchResult(doc_id="doc_2", score=12.4, content="Qdrant supports high-performance vector indexing with HNSW.", metadata={"source": "qdrant_docs.md"}),
            SearchResult(doc_id="doc_3", score=8.1, content="BM25 search ranks keyword relevance efficiently.", metadata={"source": "bm25_guide.txt"}),
        ]

        return self.reciprocal_rank_fusion(dummy_dense, dummy_sparse, top_k=top_k)
