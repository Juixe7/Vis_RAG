import os
import json
import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from qdrant_client.http import models as rest_models

from app.core.config import settings
from app.core.qdrant_client import qdrant_manager
from app.core.metrics import LatencyTracker
from app.models.schemas import DocumentChunk, SearchQuery, SearchResponse, SearchResult, HybridSearchQuery
from app.retrieval.clip_indexer import clip_indexer
from app.retrieval.bm25_search import bm25_indexer
from app.retrieval.hybrid_engine import hybrid_retriever
from app.retrieval.pdf_parser import pdf_parser
from app.engine.vllm_client import vlm_engine

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Uploads documents/images/PDFs, parses multi-page content, and dynamically indexes them into Qdrant & BM25."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    client = qdrant_manager.get_client()
    collection_name = settings.QDRANT_COLLECTION
    qdrant_manager.ensure_collection_exists(collection_name=collection_name, vector_size=512)

    indexed_chunks: List[DocumentChunk] = []
    points = []

    for file in files:
        file_id = str(uuid.uuid4())[:8]
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

        content_bytes = await file.read()
        with open(file_path, "wb") as f:
            f.write(content_bytes)

        # Handle PDF multi-page parsing
        if file.filename.lower().endswith(".pdf"):
            try:
                pdf_chunks = pdf_parser.parse_and_index_pdf(file_path, collection_name=collection_name)
                indexed_chunks.extend(pdf_chunks)
                continue
            except Exception as e:
                logger.error(f"Error parsing PDF '{file.filename}': {e}")

        text_content = ""
        image_path = None

        if file.filename.lower().endswith((".txt", ".md", ".json", ".py")):
            text_content = content_bytes.decode("utf-8", errors="ignore")
        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = file_path
            text_content = f"Image file: {file.filename}"
        else:
            text_content = f"Uploaded document file: {file.filename}"

        # Compute CLIP embedding
        if image_path:
            dense_vec = clip_indexer.get_image_embedding(image_path)
        else:
            dense_vec = clip_indexer.get_text_embedding(text_content)

        chunk = DocumentChunk(
            id=f"upload_{file_id}",
            content=text_content,
            image_path=image_path,
            dense_vector=dense_vec,
            metadata={"filename": file.filename, "size_bytes": len(content_bytes)}
        )
        indexed_chunks.append(chunk)

        point_id = abs(hash(file_id)) % (10**9)
        points.append(
            rest_models.PointStruct(
                id=point_id,
                vector=dense_vec,
                payload={
                    "doc_id": chunk.id,
                    "content": text_content,
                    "image_path": image_path,
                    "metadata": chunk.metadata
                }
            )
        )

    if points:
        client.upsert(collection_name=collection_name, points=points)
    if indexed_chunks:
        bm25_indexer.add_documents(indexed_chunks)

    return {
        "status": "success",
        "message": f"Successfully processed and indexed {len(files)} file(s).",
        "indexed_files": [f.filename for f in files]
    }

@router.post("/search", response_model=SearchResponse)
async def search_endpoint(query_req: HybridSearchQuery):
    """HTTP endpoint for Hybrid Retrieval (CLIP + BM25 + RRF)."""
    results = await hybrid_retriever.hybrid_search_async(
        query=query_req.query,
        image_input=query_req.image_base64,
        top_k=query_req.top_k,
        alpha=query_req.alpha,
        rrf_k=query_req.rrf_k
    )
    return SearchResponse(query=query_req.query, results=results)

@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """High-performance WebSocket endpoint for real-time RAG streaming & latency metrics."""
    await websocket.accept()
    logger.info("WebSocket connection established for /ws/chat")

    try:
        while True:
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)

            query = data.get("query") or data.get("prompt", "")
            image_input = data.get("image_base64") or data.get("image_path")
            top_k = int(data.get("top_k", 5))
            alpha = float(data.get("alpha", 0.5))

            if not query and not image_input:
                await websocket.send_text(json.dumps({"type": "error", "error": "Empty query or image input"}))
                continue

            tracker = LatencyTracker()

            # 1. Async Non-blocking Hybrid Context Retrieval with Alpha Weight
            retrieved_docs = await hybrid_retriever.hybrid_search_async(
                query=query,
                image_input=image_input,
                top_k=top_k,
                alpha=alpha
            )

            # Send retrieved context sources header
            sources_payload = [doc.dict() for doc in retrieved_docs]
            await websocket.send_text(json.dumps({
                "type": "sources",
                "count": len(retrieved_docs),
                "sources": sources_payload
            }))

            # 2. Stream VLM Generation
            async for token in vlm_engine.stream_answer(query=query, retrieved_contexts=retrieved_docs, image_path=image_input, tracker=tracker):
                await websocket.send_text(json.dumps({
                    "type": "chunk",
                    "token": token
                }))

            # 3. Final performance metrics payload
            stats = tracker.finish()
            await websocket.send_text(json.dumps({
                "type": "end",
                "stats": {
                    "ttft_ms": stats["ttft_ms"],
                    "total_latency_ms": stats["total_latency_ms"],
                    "tokens_per_sec": stats["tokens_per_sec"],
                    "retrieved_chunks_count": len(retrieved_docs)
                }
            }))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected gracefully.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "error": str(e)}))
            await websocket.close()
        except Exception:
            pass
