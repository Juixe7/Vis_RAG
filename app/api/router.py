import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from app.models.schemas import SearchQuery, SearchResponse, RAGRequest, RAGResponse
from app.retrieval.hybrid import HybridRetriever
from app.engine.driver import LLMStreamDriver

router = APIRouter()
retriever = HybridRetriever()
engine_driver = LLMStreamDriver()

@router.post("/search", response_model=SearchResponse)
async def hybrid_search(query_req: SearchQuery):
    """HTTP endpoint for Hybrid Vector + BM25 + RRF Retrieval."""
    try:
        results = await retriever.search(
            query=query_req.query,
            image_base64=query_req.image_base64,
            top_k=query_req.top_k
        )
        return SearchResponse(query=query_req.query, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/rag")
async def rag_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming RAG inference."""
    await websocket.accept()
    try:
        while True:
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            prompt = data.get("prompt", "")
            image_base64 = data.get("image_base64")
            top_k = data.get("top_k", 5)

            # 1. Retrieve context
            retrieved_docs = await retriever.search(prompt, image_base64=image_base64, top_k=top_k)
            context_text = "\n".join([doc.content for doc in retrieved_docs])
            augmented_prompt = f"Context:\n{context_text}\n\nUser Question:\n{prompt}"

            # 2. Stream answer tokens
            async for token in engine_driver.generate_stream(augmented_prompt, image_base64=image_base64):
                await websocket.send_text(json.dumps({"token": token, "type": "chunk"}))

            await websocket.send_text(json.dumps({"type": "end"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e), "type": "error"}))
        await websocket.close()
