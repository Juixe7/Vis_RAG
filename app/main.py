import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.core.config import settings
from app.core.qdrant_client import qdrant_manager
from scripts.seed_demo import seed_database

app = FastAPI(
    title="Vision RAG Engine API",
    description="High-performance Multimodal & Vision RAG engine supporting hybrid search & streaming LLM/VLM inference.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Auto-seeds demo documents and initializes CLIP + BM25 indexes on startup."""
    try:
        seed_database()
    except Exception as e:
        print(f"Startup seeding notice: {e}")

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    qdrant_status = qdrant_manager.verify_connection()
    return {
        "status": "online",
        "engine": settings.LLM_PROVIDER,
        "qdrant": qdrant_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
