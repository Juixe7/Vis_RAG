import os
import sys
import json
import time
import asyncio
import numpy as np
import websockets

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Ensure UTF-8 console output for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.core.qdrant_client import qdrant_manager
from app.core.metrics import LatencyTracker
from app.retrieval.hybrid_engine import HybridRetriever
from app.engine.vllm_client import vlm_engine
from scripts.seed_demo import seed_database

WS_URL = "ws://localhost:8000/api/v1/ws/chat"
TOTAL_REQUESTS = 100
CONCURRENCY = 20

TEST_PROMPTS = [
    "Explain system architecture diagram",
    "Summarize Q3 financial report revenue",
    "How does hybrid CLIP + BM25 search work?",
    "Multimodal retrieval augmented generation overview",
    "What are the benefits of Reciprocal Rank Fusion?"
]

# Ensure database is seeded for benchmarks
COLLECTION_NAME = seed_database() or "vision_rag_collection"
retriever = HybridRetriever(collection_name=COLLECTION_NAME)

async def simulate_client_request(request_id: int, prompt: str, results_list: list, semaphore: asyncio.Semaphore):
    """Simulates a single client request over WebSocket or direct engine streaming."""
    async with semaphore:
        tracker = LatencyTracker()

        try:
            # Attempt WebSocket connection if available
            async with websockets.connect(WS_URL, open_timeout=1.0) as ws:
                await ws.send(json.dumps({"query": prompt, "top_k": 3}))

                async for message in ws:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    if msg_type == "chunk":
                        tracker.record_token()
                    elif msg_type == "end":
                        stats = data.get("stats", {})
                        stats["success"] = True
                        results_list.append(stats)
                        return
        except Exception:
            # Direct in-process engine streaming benchmark simulation
            retrieved_docs = retriever.hybrid_search(query=prompt, top_k=3)
            tokens = []
            async for token in vlm_engine.stream_answer(query=prompt, retrieved_contexts=retrieved_docs, tracker=tracker):
                tokens.append(token)

            stats = tracker.finish()
            stats["success"] = True
            results_list.append(stats)

async def run_benchmark():
    print("\n" + "=" * 65)
    print("🚀 VISION RAG ENGINE BENCHMARK SUITE")
    print(f"Simulating {CONCURRENCY} Concurrent Clients | {TOTAL_REQUESTS} Total Requests")
    print("=" * 65)

    # Warmup
    print("Warming up CLIP embedding model & hybrid retriever...")
    retriever.hybrid_search("warmup query", top_k=1)
    print("Warmup complete. Starting concurrent benchmark execution...")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    results_list = []
    start_time = time.perf_counter()

    tasks = [
        simulate_client_request(i + 1, TEST_PROMPTS[i % len(TEST_PROMPTS)], results_list, semaphore)
        for i in range(TOTAL_REQUESTS)
    ]
    await asyncio.gather(*tasks)

    total_duration = time.perf_counter() - start_time

    # Calculate Performance Statistics
    ttfts = [r["ttft_ms"] for r in results_list if "ttft_ms" in r]
    latencies = [r["total_latency_ms"] for r in results_list if "total_latency_ms" in r]
    throughputs = [r["tokens_per_sec"] for r in results_list if "tokens_per_sec" in r]
    success_count = sum(1 for r in results_list if r.get("success"))

    print("\n" + "=" * 65)
    print("📊 SUMMARY PERFORMANCE REPORT")
    print("=" * 65)
    print(f" Total Requests Completed   : {len(results_list)} / {TOTAL_REQUESTS}")
    print(f" Success Rate              : {(success_count / TOTAL_REQUESTS) * 100:.1f}%")
    print(f" Total Benchmark Duration  : {total_duration:.2f} s")
    print("-" * 65)
    print(f" Mean Time-To-First-Token  : {np.mean(ttfts):.2f} ms")
    print(f" Median TTFT (P50)         : {np.median(ttfts):.2f} ms")
    print(f" P95 TTFT                  : {np.percentile(ttfts, 95):.2f} ms")
    print(f" P99 TTFT                  : {np.percentile(ttfts, 99):.2f} ms")
    print("-" * 65)
    print(f" Mean Total Latency        : {np.mean(latencies):.2f} ms")
    print(f" P95 Total Latency         : {np.percentile(latencies, 95):.2f} ms")
    print(f" Average Throughput        : {np.mean(throughputs):.2f} tok/s")
    print("=" * 65)

    # Plot Chart Results
    generate_benchmark_chart(ttfts, latencies)

def generate_benchmark_chart(ttfts: list, latencies: list):
    if plt is None:
        print("\n⚠️ Matplotlib not installed; skipping chart generation.")
        return

    try:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.hist(ttfts, bins=15, color="#06b6d4", edgecolor="#0891b2", alpha=0.85)
        plt.title("TTFT Distribution (ms)", fontsize=12, fontweight="bold", pad=10)
        plt.xlabel("TTFT (ms)")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        plt.subplot(1, 2, 2)
        plt.plot(latencies, color="#3b82f6", marker="o", markersize=3, linestyle="-", linewidth=1.5)
        plt.title("Request Latency Timeline (ms)", fontsize=12, fontweight="bold", pad=10)
        plt.xlabel("Request Index")
        plt.ylabel("Total Latency (ms)")
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        chart_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "results.png"))
        plt.savefig(chart_path, dpi=200)
        plt.close()
        print(f"\n📈 Benchmark chart successfully generated: {chart_path}")
    except Exception as e:
        print(f"\n⚠️ Could not generate chart plot: {e}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
