import time
from typing import Dict, Any, Optional

class LatencyTracker:
    """Tracker for recording request start time, Time-To-First-Token (TTFT), total generation duration, and throughput."""

    def __init__(self):
        self.start_time: float = time.perf_counter()
        self.first_token_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.ttft_ms: float = 0.0
        self.total_latency_ms: float = 0.0
        self.token_count: int = 0
        self.tokens_per_sec: float = 0.0

    def record_first_token(self):
        """Records the exact timestamp when the first generation token is emitted."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
            self.ttft_ms = round((self.first_token_time - self.start_time) * 1000, 2)

    def record_token(self):
        """Increments token count and records first token time if not set."""
        if self.first_token_time is None:
            self.record_first_token()
        self.token_count += 1

    def finish(self) -> Dict[str, Any]:
        """Finalizes latency measurements and calculates throughput metrics."""
        self.end_time = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = self.end_time
            self.ttft_ms = round((self.first_token_time - self.start_time) * 1000, 2)

        duration = self.end_time - self.start_time
        self.total_latency_ms = round(duration * 1000, 2)
        gen_duration = (self.end_time - self.first_token_time) if self.first_token_time else duration
        if gen_duration > 0 and self.token_count > 0:
            self.tokens_per_sec = round(self.token_count / gen_duration, 2)
        else:
            self.tokens_per_sec = 0.0

        return self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Returns performance metrics dictionary."""
        return {
            "ttft_ms": self.ttft_ms,
            "total_latency_ms": self.total_latency_ms,
            "token_count": self.token_count,
            "tokens_per_sec": self.tokens_per_sec
        }
