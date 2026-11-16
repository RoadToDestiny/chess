"""
Tests for Stage 2: PGN reading, validation, time control parsing, and filtering.
Covers functionality of Backend 1 and Backend 2.
"""

import io
import chess
import chess.pgn
import pytest

from chess_error_prediction.data.pgn_reader import parse_time_control, read_games, open_pgn_file
from chess_error_prediction.data.validation import validate_game
from chess_error_prediction.data.game_filter import FilterCriteria, game_matches


SAMPLE_VALID_PGN = """[Event "Rated Classical"]
[Site "https://lichess.org/test1"]
[Date "2026.10.22"]
[Round "1"]
[White "PlayerA"]
[Black "PlayerB"]
[Result "1-0"]
[WhiteElo "1950"]
[BlackElo "1900"]
[TimeControl "600+5"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 5. d3 d6 6. c3 a6 7. Bb3 Ba7 8. Nbd2 O-O 9. Re1 Re8 10. Nf1 Be6 1-0
"""


def test_open_pgn_file_plain_and_compressed(tmp_path):
    # Plain PGN
    pgn_file = tmp_path / "test.pgn"
    pgn_file.write_text(SAMPLE_VALID_PGN, encoding="utf-8")
    with open_pgn_file(pgn_file) as f:
        games = list(read_games(f))
        assert len(games) == 1
        assert games[0].headers["Event"] == "Rated Classical"

    # Gzip PGN
    import gzip
    gz_file = tmp_path / "test.pgn.gz"
    with gzip.open(gz_file, "wt", encoding="utf-8") as f:
        f.write(SAMPLE_VALID_PGN)
    with open_pgn_file(gz_file) as f:
        games = list(read_games(f))
        assert len(games) == 1
        assert games[0].headers["White"] == "PlayerA"

    # Zstandard PGN
    import zstandard
    zst_file = tmp_path / "test.pgn.zst"
    cctx = zstandard.ZstdCompressor()
    compressed = cctx.compress(SAMPLE_VALID_PGN.encode("utf-8"))
    zst_file.write_bytes(compressed)
    with open_pgn_file(zst_file) as f:
        games = list(read_games(f))
        assert len(games) == 1
        assert games[0].headers["BlackElo"] == "1900"

SAMPLE_INVALID_PGN = """[Event "No headers"]
[Result "*"]

1. e4 e5 *
"""


def test_time_control_parsing():
    base, inc = parse_time_control("600+0")
    assert base == 600
    assert inc == 0

    base, inc = parse_time_control("900+10")
    assert base == 900
    assert inc == 10

    with pytest.raises(ValueError):
        parse_time_control("invalid")


def test_clock_seconds_parsing():
    from chess_error_prediction.data.pgn_reader import parse_clock_seconds
    assert parse_clock_seconds("[%clk 0:10:00]") == 600.0
    assert parse_clock_seconds("[%clk 0:01:42]") == 102.0
    assert parse_clock_seconds("[%clk 12:34]") == 754.0
    assert parse_clock_seconds("") is None
    assert parse_clock_seconds("plain comment without clock") is None


def test_game_validation_valid_and_invalid():
    game_valid = chess.pgn.read_game(io.StringIO(SAMPLE_VALID_PGN))
    assert game_valid is not None
    is_valid, reasons = validate_game(game_valid)
    assert is_valid, f"Validation failed: {reasons}"

    game_invalid = chess.pgn.read_game(io.StringIO(SAMPLE_INVALID_PGN))
    assert game_invalid is not None
    is_valid, reasons = validate_game(game_invalid)
    assert not is_valid
    assert len(reasons) > 0


def test_filter_criteria_matching():
    game = chess.pgn.read_game(io.StringIO(SAMPLE_VALID_PGN))
    assert game is not None

    criteria_pass = FilterCriteria(min_elo=1600, min_base_seconds=600, min_full_moves=10)
    assert game_matches(game, criteria_pass)

    criteria_fail_elo = FilterCriteria(min_elo=2200)
    assert not game_matches(game, criteria_fail_elo)

    criteria_fail_time = FilterCriteria(min_base_seconds=900)
    assert not game_matches(game, criteria_fail_time)

    rapid_boundary = chess.pgn.read_game(io.StringIO(SAMPLE_VALID_PGN))
    rapid_boundary.headers["TimeControl"] = "900+20"
    assert not game_matches(rapid_boundary, FilterCriteria())

    # Bot rejection test
    game_bot = chess.pgn.read_game(io.StringIO(SAMPLE_VALID_PGN))
    game_bot.headers["WhiteTitle"] = "BOT"
    assert not game_matches(game_bot, criteria_pass)

    # Variant rejection test (e.g. Chess960 or Atomic)
    game_variant = chess.pgn.read_game(io.StringIO(SAMPLE_VALID_PGN))
    game_variant.headers["Variant"] = "Chess960"
    assert not game_matches(game_variant, criteria_pass)


def test_depersonalize_game():
    from chess_error_prediction.data.game_filter import depersonalize_game
    game = chess.pgn.read_game(io.StringIO(SAMPLE_VALID_PGN))
    assert game.headers["White"] == "PlayerA"
    assert game.headers["Black"] == "PlayerB"

    anon = depersonalize_game(game)
    assert anon.headers["White"].startswith("Anon_W_")
    assert anon.headers["Black"].startswith("Anon_B_")
    assert anon.headers["WhiteElo"] == "1950"
    assert anon.headers["TimeControl"] == "600+5"
