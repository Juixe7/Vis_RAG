import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from app.models.schemas import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
INDEX_FILE = os.path.join(INDEX_DIR, "bm25_corpus.json")

class BM25Indexer:
    """Sparse keyword search indexer using BM25Okapi algorithm with incremental updates and disk persistence."""

    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path or INDEX_FILE
        self.documents: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []
        self.load_from_disk()

    def tokenize(self, text: str) -> List[str]:
        """Simple, clean regex tokenization converting text to lowercase alphanumeric tokens."""
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    def _rebuild_index(self):
        """Rebuilds the BM25Okapi index from the current corpus."""
        self.corpus_tokens = [self.tokenize(doc.content) for doc in self.documents]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def save_to_disk(self):
        """Persists the BM25 corpus to disk as JSON."""
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            data = []
            for doc in self.documents:
                data.append({
                    "id": doc.id,
                    "content": doc.content,
                    "image_path": doc.image_path,
                    "metadata": doc.metadata
                })
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.documents)} BM25 documents to {self.persistence_path}")
        except Exception as e:
            logger.warning(f"Failed to persist BM25 index: {e}")

    def load_from_disk(self):
        """Loads BM25 corpus from disk if file exists."""
        if not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded_docs = []
            for item in data:
                loaded_docs.append(DocumentChunk(
                    id=item["id"],
                    content=item["content"],
                    image_path=item.get("image_path"),
                    metadata=item.get("metadata", {})
                ))
            self.documents = loaded_docs
            self._rebuild_index()
            logger.info(f"Loaded {len(self.documents)} BM25 documents from {self.persistence_path}")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index from disk: {e}")

    def add_documents(self, documents: List[DocumentChunk]):
        """Incrementally adds new documents to the BM25 index, updating existing ones by ID."""
        if not documents:
            return

        doc_map = {doc.id: idx for idx, doc in enumerate(self.documents)}
        for new_doc in documents:
            if new_doc.id in doc_map:
                self.documents[doc_map[new_doc.id]] = new_doc
            else:
                doc_map[new_doc.id] = len(self.documents)
                self.documents.append(new_doc)

        self._rebuild_index()
        self.save_to_disk()

    def index_documents(self, documents: List[DocumentChunk]):
        """Adds documents incrementally while persisting to disk."""
        self.add_documents(documents)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Searches the indexed corpus using BM25 and returns top_k SearchResult items."""
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx]
                results.append(
                    SearchResult(
                        doc_id=doc.id,
                        score=float(scores[idx]),
                        content=doc.content,
                        image_path=doc.image_path,
                        metadata=doc.metadata
                    )
                )
        return results

bm25_indexer = BM25Indexer()
