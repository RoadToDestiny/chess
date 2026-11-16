"""Reproducible counters for the PGN-to-dataset analysis pipeline."""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict
import tracemalloc


@dataclass
class BenchmarkMetrics:
    """Counters and timings collected during one dataset build."""

    games_read: int = 0
    moves_read: int = 0
    valid_rows: int = 0
    excluded_pre_move_mates: int = 0
    saved_post_move_mate_errors: int = 0
    excluded_other_terminal: int = 0
    stockfish_calls: int = 0
    stockfish_seconds: float = 0.0
    pgn_seconds: float = 0.0
    feature_seconds: float = 0.0
    dataset_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    _stage_started: Dict[str, float] = field(default_factory=dict, repr=False)

    def start_stage(self, name: str) -> None:
        self._stage_started[name] = perf_counter()

    def stop_stage(self, name: str) -> None:
        started = self._stage_started.pop(name, None)
        if started is None:
            raise RuntimeError(f"Benchmark stage was not started: {name}")
        elapsed = perf_counter() - started
        if name == "pgn":
            self.pgn_seconds += elapsed
        elif name == "features":
            self.feature_seconds += elapsed
        elif name == "dataset":
            self.dataset_seconds += elapsed

    def record_stockfish_call(self, elapsed: float) -> None:
        self.stockfish_calls += 1
        self.stockfish_seconds += elapsed

    def refresh_peak_memory(self) -> None:
        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            self.peak_memory_mb = max(self.peak_memory_mb, peak / (1024 * 1024))

    def to_dict(self) -> Dict[str, float]:
        self.refresh_peak_memory()
        return {
            "games_read": self.games_read,
            "moves_read": self.moves_read,
            "valid_rows": self.valid_rows,
            "excluded_pre_move_mates": self.excluded_pre_move_mates,
            "saved_post_move_mate_errors": self.saved_post_move_mate_errors,
            "excluded_other_terminal": self.excluded_other_terminal,
            "stockfish_calls": self.stockfish_calls,
            "stockfish_seconds": round(self.stockfish_seconds, 6),
            "pgn_seconds": round(self.pgn_seconds, 6),
            "feature_seconds": round(self.feature_seconds, 6),
            "dataset_seconds": round(self.dataset_seconds, 6),
            "peak_memory_mb": round(self.peak_memory_mb, 3),
        }
