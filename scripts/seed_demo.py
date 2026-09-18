import os
import sys
import glob
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

# Ensure UTF-8 output encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.core.qdrant_client import qdrant_manager
from app.models.schemas import DocumentChunk, SearchResult
from app.retrieval.clip_indexer import clip_indexer
from app.retrieval.bm25_search import bm25_indexer
from app.retrieval.hybrid_engine import hybrid_retriever, HybridRetriever

COLLECTION_NAME = settings.QDRANT_COLLECTION

def create_sample_docs_if_empty(sample_dir: str):
    """Ensures sample documents and images exist in data/sample_docs/."""
    os.makedirs(sample_dir, exist_ok=True)
    sample_files = glob.glob(os.path.join(sample_dir, "*"))

    if not sample_files:
        print(f"Creating sample document and image pairs in {sample_dir}...")
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            Image = None

        docs = [
            {
                "text_file": "arch_diagram.txt",
                "img_file": "arch_diagram.png",
                "content": "Vision RAG Engine System Architecture Diagram: Microservice with FastAPI backend, Qdrant vector database, vLLM / Ollama streaming engine, and Hybrid RRF Retriever.",
                "color": (73, 109, 137),
                "label": "Architecture"
            },
            {
                "text_file": "financial_report.txt",
                "img_file": "financial_report.png",
                "content": "Q3 Financial Report & Revenue Analysis: Annual recurring revenue grew by 45% YoY with an operating margin of 28% across enterprise sales.",
                "color": (150, 75, 75),
                "label": "Financials"
            },
            {
                "text_file": "vision_rag_paper.txt",
                "img_file": "vision_rag_paper.png",
                "content": "Multimodal Vision RAG Paper: Combining dense CLIP visual embeddings and sparse BM25 text keyword retrieval using Reciprocal Rank Fusion.",
                "color": (50, 150, 50),
                "label": "Vision RAG"
            }
        ]

        for item in docs:
            txt_path = os.path.join(sample_dir, item["text_file"])
            img_path = os.path.join(sample_dir, item["img_file"])

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(item["content"])

            if Image:
                img = Image.new("RGB", (300, 300), color=item["color"])
                draw = ImageDraw.Draw(img)
                draw.text((20, 20), item["label"], fill=(255, 255, 255))
                img.save(img_path)

def seed_database():
    print("=" * 60)
    print("SEEDING DEMO DATA FOR VISION RAG ENGINE")
    print("=" * 60)

    sample_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs"))
    create_sample_docs_if_empty(sample_dir)

    # Load all sample text files
    txt_files = sorted(glob.glob(os.path.join(sample_dir, "*.txt")))
    if not txt_files:
        print(f"No text files found in {sample_dir}")
        return

    client = qdrant_manager.get_client()

    # Create collection if it doesn't exist (vector dimension 512 for clip-ViT-B-32)
    vector_dim = 512
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating Qdrant collection '{COLLECTION_NAME}' (dim={vector_dim})...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest_models.VectorParams(
                size=vector_dim,
                distance=rest_models.Distance.COSINE
            )
        )

    chunks: List[DocumentChunk] = []
    points = []

    print(f"\nProcessing {len(txt_files)} sample document pairs...")
    for idx, txt_path in enumerate(txt_files, start=1):
        filename = os.path.basename(txt_path)
        base_name = os.path.splitext(filename)[0]

        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        img_path = os.path.join(sample_dir, f"{base_name}.png")
        if not os.path.exists(img_path):
            img_path = None

        print(f" [{idx}] Indexing: {filename} | Image attached: {os.path.basename(img_path) if img_path else 'None'}")

        # Compute CLIP dense embedding (combining text and image context)
        dense_vector = clip_indexer.get_text_embedding(content)

        chunk_id = f"doc_{idx}"
        chunk = DocumentChunk(
            id=chunk_id,
            content=content,
            image_path=img_path,
            dense_vector=dense_vector,
            metadata={"filename": filename, "source_dir": sample_dir}
        )
        chunks.append(chunk)

        points.append(
            rest_models.PointStruct(
                id=idx,
                vector=dense_vector,
                payload={
                    "doc_id": chunk_id,
                    "content": content,
                    "image_path": img_path,
                    "metadata": chunk.metadata
                }
            )
        )

    # 1. Upsert points into Qdrant
    print(f"\nUpserting {len(points)} vector points into Qdrant ('{COLLECTION_NAME}')...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    # 2. Index chunks into BM25
    print("Indexing text chunks into BM25 sparse index...")
    bm25_indexer.index_documents(chunks)

    print("\nSUCCESS: SEEDING COMPLETE!")
    return COLLECTION_NAME

def run_test_queries(collection_name: str):
    print("\n" + "=" * 60)
    print("TESTING HYBRID RETRIEVAL (CLIP + BM25 + RRF)")
    print("=" * 60)

    retriever = HybridRetriever(collection_name=collection_name)

    test_queries = [
        "system architecture diagram",
        "financial report",
        "vision RAG paper"
    ]

    for q in test_queries:
        print(f"\nQuery: '{q}'")
        print("-" * 50)
        results: List[SearchResult] = retriever.hybrid_search(query=q, top_k=3, rrf_k=60)

        for rank, res in enumerate(results, start=1):
            print(f" Rank {rank} | RRF Score: {res.score:.6f} | Doc ID: {res.doc_id}")
            print(f" Content: {res.content[:90]}...")
            if res.image_path:
                print(f" Image: {os.path.basename(res.image_path)}")
            print()

if __name__ == "__main__":
    col_name = seed_database()
    if col_name:
        run_test_queries(col_name)
