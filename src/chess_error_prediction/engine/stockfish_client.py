"""
Stockfish UCI engine client and lifecycle management.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional
import chess
import chess.engine


class StockfishError(RuntimeError):
    """Raised when Stockfish cannot be started or queried."""


@dataclass(frozen=True)
class StockfishConfig:
    """Configuration parameters for Stockfish analysis."""
    executable: str = "stockfish"
    depth: int = 18
    timeout: float = 30.0
    threads: Optional[int] = None
    hash_mb: Optional[int] = None
    mate_score_cp: int = 100000
    expected_version: Optional[str] = None

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("Stockfish depth must be at least 1.")
        if self.timeout <= 0:
            raise ValueError("Stockfish timeout must be positive.")
        if self.mate_score_cp <= 0:
            raise ValueError("Mate score in centipawns must be positive.")

    @classmethod
    def from_environment(cls, **overrides: Any) -> "StockfishConfig":
        """Build configuration using ``STOCKFISH_PATH`` when it is defined."""
        executable = overrides.pop(
            "executable",
            os.environ.get("STOCKFISH_PATH", "stockfish"),
        )
        return cls(executable=executable, **overrides)


class StockfishEngine:
    """UCI wrapper providing a consistent White-POV evaluation interface."""

    def __init__(self, config: Optional[StockfishConfig] = None) -> None:
        self.config = config or StockfishConfig()
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def start(self) -> "StockfishEngine":
        if self._engine is not None:
            return self
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(
                self.config.executable,
                timeout=self.config.timeout,
            )
            options: Dict[str, Any] = {}
            if self.config.threads is not None:
                options["Threads"] = self.config.threads
            if self.config.hash_mb is not None:
                options["Hash"] = self.config.hash_mb
            if options:
                self._engine.configure(options)
        except (OSError, TimeoutError, chess.engine.EngineError) as exc:
            self._engine = None
            raise StockfishError(
                f"Unable to start Stockfish executable '{self.config.executable}'. "
                "Ensure Stockfish is installed and in PATH, or specify full path."
            ) from exc
        return self

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except chess.engine.EngineError as exc:
                raise StockfishError("Stockfish failed while shutting down.") from exc
            finally:
                self._engine = None

    def __enter__(self) -> "StockfishEngine":
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    @property
    def engine_id(self) -> Dict[str, str]:
        if self._engine is None:
            raise StockfishError("Stockfish is not started.")
        return dict(self._engine.id)

    def analyze(self, board: chess.Board) -> Dict[str, Any]:
        """
        Analyzes board position and returns centipawn score strictly from White's POV.
        """
        if self._engine is None:
            raise StockfishError("Stockfish is not started.")
        try:
            info = self._engine.analyse(
                board,
                chess.engine.Limit(depth=self.config.depth),
            )
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
            raise StockfishError("Stockfish failed while analyzing position.") from exc

        score = info["score"].pov(chess.WHITE)
        score_cp = score.score(mate_score=self.config.mate_score_cp)
        return {
            "score_cp": float(score_cp) if score_cp is not None else None,
            "is_mate": score.is_mate(),
            "mate_in": score.mate(),
            "depth": info.get("depth", self.config.depth),
        }

    def verify(self) -> Dict[str, str]:
        """Verifies startup by analyzing initial position."""
        self.start()
        self.analyze(chess.Board())
        engine_id = self.engine_id
        if (
            self.config.expected_version is not None
            and engine_id.get("version") != self.config.expected_version
        ):
            raise StockfishError(
                f"Unexpected Stockfish version: {engine_id.get('version', '<unknown>')!r}; "
                f"expected {self.config.expected_version!r}."
            )
        return engine_id
