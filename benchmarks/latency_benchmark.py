import time
import asyncio
import httpx

async def benchmark_health_check():
    url = "http://localhost:8000/health"
    print(f"Connecting to {url}...")
    start_time = time.perf_counter()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0)
            elapsed = (time.perf_counter() - start_time) * 1000
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
            print(f"Latency: {elapsed:.2f} ms")
    except Exception as e:
        print(f"Benchmark error: {e}")

if __name__ == "__main__":
    asyncio.run(benchmark_health_check())
