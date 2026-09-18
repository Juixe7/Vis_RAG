import os
import uuid
import logging
from typing import List
import fitz  # PyMuPDF
from qdrant_client.http import models as rest_models

from app.core.config import settings
from app.core.qdrant_client import qdrant_manager
from app.models.schemas import DocumentChunk
from app.retrieval.clip_indexer import clip_indexer
from app.retrieval.bm25_search import bm25_indexer

logger = logging.getLogger(__name__)

PAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads", "pdf_pages"))
os.makedirs(PAGES_DIR, exist_ok=True)

class PDFDocumentParser:
    """Multi-page PDF document parser using PyMuPDF (fitz) to extract text & visual page images."""

    def parse_and_index_pdf(self, pdf_path: str, collection_name: str = None) -> List[DocumentChunk]:
        """Parses multi-page PDF into page image and text chunks, computes CLIP embeddings, and indexes into Qdrant & BM25."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        target_collection = collection_name or settings.QDRANT_COLLECTION
        qdrant_manager.ensure_collection_exists(collection_name=target_collection, vector_size=512)

        doc = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(filename)[0]

        chunks: List[DocumentChunk] = []
        points = []

        logger.info(f"Parsing PDF document: '{filename}' ({len(doc)} pages)...")

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text").strip()

            # Render page to PNG image
            pix = page.get_pixmap(dpi=150)
            page_img_filename = f"{base_name}_page_{page_num + 1}_{str(uuid.uuid4())[:6]}.png"
            page_img_path = os.path.join(PAGES_DIR, page_img_filename)
            pix.save(page_img_path)

            content = page_text if page_text else f"PDF Document Page {page_num + 1} of {filename}"

            # Compute CLIP dense embedding for the page image
            try:
                dense_vec = clip_indexer.get_image_embedding(page_img_path)
            except Exception:
                dense_vec = clip_indexer.get_text_embedding(content)

            chunk_id = f"pdf_{base_name}_p{page_num + 1}"
            chunk = DocumentChunk(
                id=chunk_id,
                content=content,
                image_path=page_img_path,
                dense_vector=dense_vec,
                metadata={
                    "filename": filename,
                    "page_number": page_num + 1,
                    "total_pages": len(doc),
                    "type": "pdf_page"
                }
            )
            chunks.append(chunk)

            point_id = abs(hash(f"{filename}_p{page_num + 1}")) % (10**9)
            points.append(
                rest_models.PointStruct(
                    id=point_id,
                    vector=dense_vec,
                    payload={
                        "doc_id": chunk.id,
                        "content": content,
                        "image_path": page_img_path,
                        "metadata": chunk.metadata
                    }
                )
            )

        doc.close()

        # Upsert vector points to Qdrant
        client = qdrant_manager.get_client()
        if points:
            client.upsert(collection_name=target_collection, points=points)

        # Index chunks in BM25
        if chunks:
            bm25_indexer.add_documents(chunks)

        logger.info(f"Successfully indexed {len(chunks)} page chunk(s) from PDF '{filename}'.")
        return chunks

pdf_parser = PDFDocumentParser()
