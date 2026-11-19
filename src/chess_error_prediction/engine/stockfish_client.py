"""
Stockfish UCI engine client and lifecycle management.
"""

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Dict, Optional
import chess
import chess.engine


class StockfishError(RuntimeError):
    """Raised when Stockfish cannot be started or queried."""


def find_default_executable() -> str:
    """
    Locates Stockfish executable in:
    1. STOCKFISH_PATH environment variable
    2. Local project bin/ directory (e.g. bin/stockfish-windows-x86-64-universal.exe)
    3. System PATH ('stockfish')
    """
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Local project bin directory
    project_root = Path(__file__).resolve().parents[3]
    bin_dir = project_root / "bin"
    candidates = [
        bin_dir / "stockfish-windows-x86-64-universal.exe",
        bin_dir / "stockfish.exe",
        bin_dir / "stockfish",
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)

    return "stockfish"


@dataclass(frozen=True)
class StockfishConfig:
    """
    Configuration parameters for Stockfish analysis.
    Fixed parameters for scientific reproducibility:
    - version: Stockfish 19
    - depth: 18 plies
    - threads: 1 (deterministic single-thread evaluation)
    - hash: 128 MB
    - mate_score_cp: 100000
    """
    executable: str = ""
    depth: int = 18
    timeout: float = 30.0
    threads: int = 1
    hash_mb: int = 128
    mate_score_cp: int = 100000
    expected_version: Optional[str] = "Stockfish 19"

    def __post_init__(self) -> None:
        if not self.executable:
            object.__setattr__(self, "executable", find_default_executable())
        if self.depth < 1:
            raise ValueError("Stockfish depth must be at least 1.")
        if self.timeout <= 0:
            raise ValueError("Stockfish timeout must be positive.")
        if self.threads < 1:
            raise ValueError("Stockfish threads must be at least 1.")
        if self.hash_mb < 1:
            raise ValueError("Stockfish hash_mb must be at least 1.")
        if self.mate_score_cp <= 0:
            raise ValueError("Mate score in centipawns must be positive.")

    @classmethod
    def from_environment(cls, **overrides: Any) -> "StockfishConfig":
        """Build configuration using ``STOCKFISH_PATH`` or local bin when defined."""
        executable = overrides.pop(
            "executable",
            find_default_executable(),
        )
        return cls(executable=executable, **overrides)

    @classmethod
    def from_config(cls, **overrides: Any) -> "StockfishConfig":
        """Build configuration from config/config.yaml under 'stockfish' section with overrides."""
        try:
            from chess_error_prediction.config_loader import get_stockfish_config
            sf_cfg = get_stockfish_config()
        except Exception:
            sf_cfg = {}
        executable = overrides.pop("executable", sf_cfg.get("executable") or find_default_executable())
        depth = overrides.pop("depth", int(sf_cfg.get("depth", 18)))
        timeout = overrides.pop("timeout", float(sf_cfg.get("timeout", 30.0)))
        threads = overrides.pop("threads", int(sf_cfg.get("threads", 1)))
        hash_mb = overrides.pop("hash_mb", int(sf_cfg.get("hash_mb", 128)))
        mate_score_cp = overrides.pop("mate_score_cp", int(sf_cfg.get("mate_score_cp", 100000)))
        expected_version = overrides.pop("expected_version", sf_cfg.get("version", "Stockfish 19"))
        return cls(
            executable=executable,
            depth=depth,
            timeout=timeout,
            threads=threads,
            hash_mb=hash_mb,
            mate_score_cp=mate_score_cp,
            expected_version=expected_version,
            **overrides,
        )


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

    def analyze(self, board: chess.Board, game: Any = None) -> Dict[str, Any]:
        """
        Analyzes board position and returns centipawn score strictly from White's POV.
        Optionally accepts game token to inform UCI protocol of game boundaries.
        """
        if self._engine is None:
            raise StockfishError("Stockfish is not started.")
        try:
            info = self._engine.analyse(
                board,
                chess.engine.Limit(depth=self.config.depth),
                game=game,
            )
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
            raise StockfishError("Stockfish failed while analyzing position.") from exc

        score = info["score"].pov(chess.WHITE)
        score_cp = score.score(mate_score=self.config.mate_score_cp)
        # Apply technical clipping [-2500, +2500] cp to non-mate static evaluations
        if score_cp is not None and not score.is_mate():
            score_cp = max(-2500.0, min(2500.0, float(score_cp)))

        return {
            "score_cp": float(score_cp) if score_cp is not None else None,
            "is_mate": score.is_mate(),
            "mate_in": score.mate(),
            "depth": info.get("depth", self.config.depth),
        }

    def reset_game(self) -> None:
        """
        Sends ucinewgame to Stockfish via UCI protocol and safely synchronizes with ping (isready/readyok).
        Clears transposition table between independent games.
        """
        if self._engine is not None:
            try:
                self._engine.ping()
                if hasattr(self._engine, "protocol") and hasattr(self._engine.protocol, "send_line"):
                    self._engine.protocol.send_line("ucinewgame")
                self._engine.ping()
            except Exception:
                pass

    def verify(self) -> Dict[str, str]:
        """Verifies startup by analyzing initial position."""
        self.start()
        self.analyze(chess.Board())
        engine_id = self.engine_id
        if self.config.expected_version is not None:
            name = engine_id.get("name", "")
            version = engine_id.get("version", "")
            if self.config.expected_version not in name and self.config.expected_version != version:
                raise StockfishError(
                    f"Unexpected Stockfish version: name={name!r}, version={version!r}; "
                    f"expected {self.config.expected_version!r}."
                )
        return engine_id
