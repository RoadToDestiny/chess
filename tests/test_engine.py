"""
Tests for Stockfish configuration, lifecycle, and evaluation normalization.
"""

import os
from pathlib import Path
import pytest
import chess
from chess_error_prediction.engine.stockfish_client import StockfishConfig, StockfishEngine, StockfishError
from chess_error_prediction.engine.evaluation import normalize_evaluation, EvaluationResult


def test_stockfish_config_defaults():
    cfg = StockfishConfig()
    assert cfg.depth == 18
    assert "stockfish" in cfg.executable.lower()
    assert cfg.timeout == 30.0
    assert cfg.threads == 1
    assert cfg.hash_mb == 128
    assert cfg.mate_score_cp == 100000
    assert cfg.expected_version == "Stockfish 19"


def test_stockfish_config_validation():
    with pytest.raises(ValueError):
        StockfishConfig(depth=0)
    with pytest.raises(ValueError):
        StockfishConfig(timeout=-5.0)
    with pytest.raises(ValueError):
        StockfishConfig(threads=0)
    with pytest.raises(ValueError):
        StockfishConfig(hash_mb=0)
    with pytest.raises(ValueError):
        StockfishConfig(mate_score_cp=0)


def test_stockfish_engine_real_verification():
    cfg = StockfishConfig()
    if os.path.exists(cfg.executable):
        engine = StockfishEngine(cfg)
        with engine:
            ident = engine.verify()
            assert "Stockfish 19" in ident.get("name", "")
            res = engine.analyze(chess.Board())
            assert res["score_cp"] is not None
            assert res["depth"] == 18


def test_evaluation_normalization():
    raw_eval = {"score_cp": 125.0, "is_mate": False, "mate_in": None, "depth": 18}
    norm = normalize_evaluation(raw_eval)
    assert isinstance(norm, EvaluationResult)
    assert norm.score_cp == 125.0
    assert not norm.is_mate
    assert norm.is_valid_numeric

    mate_eval = {"score_cp": 100000.0, "is_mate": True, "mate_in": 2, "depth": 18}
    norm_mate = normalize_evaluation(mate_eval)
    assert norm_mate.is_mate
    assert not norm_mate.is_valid_numeric
    assert norm_mate.mate_in == 2


def test_analyze_game_pipeline():
    import io
    import chess.pgn
    from chess_error_prediction.engine.game_analysis import analyze_game, MoveAnalysis

    cfg = StockfishConfig()
    if os.path.exists(cfg.executable):
        pgn_text = "1. e4 e5 2. Nf3 Nc6 *"
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        engine = StockfishEngine(cfg)
        with engine:
            records = list(analyze_game(game, engine, threshold_cp=200.0))
            assert len(records) == 4
            assert isinstance(records[0], MoveAnalysis)
            assert records[0].san == "e4"
            assert records[0].color == "w"
            assert records[1].san == "e5"
            assert records[1].color == "b"
            # Verify continuous evaluation caching (after ply 1 == before ply 2)
            assert records[0].evaluation_after.score_cp == records[1].evaluation_before.score_cp
