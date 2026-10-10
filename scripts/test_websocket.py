import os
import sys
import json
import asyncio
import websockets

# Ensure UTF-8 output encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

WS_URL = "ws://localhost:8000/api/v1/ws/chat"

async def test_websocket_chat(query: str = "Explain system architecture diagram"):
    print("=" * 60)
    print("⚡ TESTING WEBSOCKET CHAT & STREAMING TTFT LATENCY METRICS")
    print(f"Connecting to: {WS_URL}")
    print("=" * 60)

    try:
        async with websockets.connect(WS_URL) as ws:
            payload = {
                "query": query,
                "top_k": 3
            }
            print(f"\nSending payload: query='{query}'...")
            await ws.send(json.dumps(payload))

            print("\nStreaming response: ", end="", flush=True)

            sources = []
            async for message in ws:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "sources":
                    sources = data.get("sources", [])
                elif msg_type == "chunk":
                    token = data.get("token", "")
                    print(token, end="", flush=True)
                elif msg_type == "end":
                    stats = data.get("stats", {})
                    print("\n\n" + "=" * 60)
                    print("📊 PERFORMANCE STATS & METRICS")
                    print("=" * 60)
                    print(f" Time-To-First-Token (TTFT): {stats.get('ttft_ms')} ms")
                    print(f" Total Generation Latency : {stats.get('total_latency_ms')} ms")
                    print(f" Tokens Per Second        : {stats.get('tokens_per_sec')} tok/s")
                    print(f" Retrieved Context Chunks  : {stats.get('retrieved_chunks_count')}")
                    print("=" * 60)
                    break
                elif msg_type == "error":
                    print(f"\n[ERROR] WebSocket error: {data.get('error')}")
                    break

    except Exception as e:
        print(f"\n❌ Could not connect to WebSocket at {WS_URL}: {e}")
        print("Tip: Make sure FastAPI server is running with 'uvicorn app.main:app --port 8000'")

if __name__ == "__main__":
    query_str = sys.argv[1] if len(sys.argv) > 1 else "Explain system architecture diagram"
    asyncio.run(test_websocket_chat(query_str))
