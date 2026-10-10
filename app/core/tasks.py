import os
import uuid
import logging
from typing import Dict, Any
from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.qdrant_client import qdrant_manager
from app.models.schemas import DocumentChunk
from app.retrieval.clip_indexer import clip_indexer
from app.retrieval.bm25_search import bm25_indexer
from qdrant_client.http import models as rest_models

logger = logging.getLogger(__name__)

@celery_app.task(name="process_and_index_document_task")
def process_and_index_document_task(file_path: str, filename: str) -> Dict[str, Any]:
    """Background Celery task to parse multi-page documents/images, extract embeddings, and index into Qdrant & BM25."""
    logger.info(f"[Celery Worker] Starting background processing for: {filename}")
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File not found: {file_path}"}

    file_id = str(uuid.uuid4())[:8]

    # Check if PDF
    if filename.lower().endswith(".pdf"):
        from app.retrieval.pdf_parser import pdf_parser
        result = pdf_parser.parse_and_index_pdf(file_path)
        return {"status": "success", "type": "pdf", "chunks": len(result)}

    # Text or Image processing
    with open(file_path, "rb") as f:
        content_bytes = f.read()

    text_content = ""
    image_path = None

    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = file_path
        text_content = f"Image document: {filename}"
        dense_vec = clip_indexer.get_image_embedding(image_path)
    else:
        text_content = content_bytes.decode("utf-8", errors="ignore")
        dense_vec = clip_indexer.get_text_embedding(text_content)

    client = qdrant_manager.get_client()
    collection_name = settings.QDRANT_COLLECTION
    qdrant_manager.ensure_collection_exists(collection_name=collection_name, vector_size=512)

    chunk = DocumentChunk(
        id=f"celery_{file_id}",
        content=text_content,
        image_path=image_path,
        dense_vector=dense_vec,
        metadata={"filename": filename, "source": "celery_worker"}
    )

    point_id = abs(hash(file_id)) % (10**9)
    point = rest_models.PointStruct(
        id=point_id,
        vector=dense_vec,
        payload={
            "doc_id": chunk.id,
            "content": text_content,
            "image_path": image_path,
            "metadata": chunk.metadata
        }
    )

    client.upsert(collection_name=collection_name, points=[point])
    bm25_indexer.index_documents([chunk])

    logger.info(f"[Celery Worker] Successfully indexed: {filename}")
    return {"status": "success", "doc_id": chunk.id, "filename": filename}
