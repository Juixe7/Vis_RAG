from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class DocumentChunk(BaseModel):
    """Schema representing a text/image document chunk to be indexed in Qdrant."""
    id: str = Field(..., description="Unique document chunk ID")
    content: str = Field(..., description="Text content of the document chunk")
    image_path: Optional[str] = Field(None, description="Local or remote path to related image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image string")
    dense_vector: Optional[List[float]] = Field(None, description="Dense vector embedding")
    sparse_indices: Optional[List[int]] = Field(None, description="Sparse vector non-zero indices")
    sparse_values: Optional[List[float]] = Field(None, description="Sparse vector non-zero values")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary (source, page, author, etc.)")

    model_config = ConfigDict(arbitrary_types_allowed=True)

class HybridSearchQuery(BaseModel):
    """Schema for Hybrid Search requests combining Dense, Sparse (BM25), and Vision queries."""
    query: str = Field(..., description="Search query string")
    image_base64: Optional[str] = Field(None, description="Base64 encoded query image for visual search")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of top search results to return")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Dense vs Sparse weight balance (1.0 = Dense only, 0.0 = Sparse only)")
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal Rank Fusion smoothing constant")

class SearchResult(BaseModel):
    """Schema representing a single search result returned from the retrieval pipeline."""
    doc_id: str = Field(..., description="ID of the matching document chunk")
    score: float = Field(..., description="Relevance score after RRF / hybrid ranking")
    content: str = Field(..., description="Text content of the result chunk")
    image_path: Optional[str] = Field(None, description="Path to associated image if present")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata attached to the chunk")

class StreamResponse(BaseModel):
    """Schema for WebSocket or SSE streaming response tokens and events."""
    token: Optional[str] = Field(None, description="Generated token chunk")
    event_type: str = Field(default="chunk", description="Event type: 'chunk', 'sources', 'end', or 'error'")
    sources: Optional[List[SearchResult]] = Field(None, description="Retrieved context sources")
    error: Optional[str] = Field(None, description="Error message if any")

# Backward Compatibility Models
class SearchQuery(BaseModel):
    query: str
    image_base64: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=50)

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

class RAGRequest(BaseModel):
    prompt: str
    image_base64: Optional[str] = None
    stream: bool = True
    top_k: int = 5

class RAGResponse(BaseModel):
    answer: str
    retrieved_docs: List[SearchResult]
