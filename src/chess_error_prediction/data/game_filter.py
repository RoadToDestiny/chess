"""
Game filtering based on time control, player ratings, and game length criteria.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple
import chess.pgn

from chess_error_prediction.data.pgn_reader import open_pgn_file, read_games, parse_time_control
from chess_error_prediction.data.validation import validate_game


@dataclass(frozen=True)
class FilterCriteria:
    """Selection criteria for filtering the research dataset."""
    min_elo: Optional[int] = 1600
    max_elo: Optional[int] = 2800
    max_elo_diff: Optional[int] = 300
    min_base_seconds: Optional[int] = 600
    max_base_seconds: Optional[int] = None
    max_estimated_seconds: Optional[int] = None
    allowed_time_controls: Optional[Tuple[str, ...]] = None
    min_full_moves: Optional[int] = 10
    max_full_moves: Optional[int] = 120
    exclude_bots: bool = True
    require_standard_variant: bool = True
    allowed_results: Tuple[str, ...] = ("1-0", "0-1", "1/2-1/2")
    exclude_terminations: Tuple[str, ...] = ("Abandoned", "Rules infraction")


@dataclass(frozen=True)
class GameFilterStats:
    """Audit statistics for PGN filtering."""
    read: int = 0
    accepted: int = 0
    rejected: int = 0
    invalid: int = 0


def _get_elo(game: chess.pgn.Game, color_key: str) -> Optional[int]:
    val = game.headers.get(color_key)
    if val and val.isdigit():
        return int(val)
    return None


def game_matches(game: chess.pgn.Game, criteria: FilterCriteria) -> bool:
    """
    Checks if a valid chess game matches all specified criteria:
    - Standard chess variant only (no Chess960, Atomic, Antichess)
    - Human vs Human only (no BOT accounts)
    - Valid time control (>= min_base_seconds)
    - Player ratings within [min_elo, max_elo]
    - Game length within [min_full_moves, max_full_moves]
    """
    if criteria.require_standard_variant:
        variant = game.headers.get("Variant", "Standard").strip()
        if variant.lower() not in ("standard", ""):
            return False

    if criteria.exclude_bots:
        if game.headers.get("WhiteTitle") == "BOT" or game.headers.get("BlackTitle") == "BOT":
            return False

    time_control = game.headers.get("TimeControl", "")
    if criteria.allowed_time_controls is not None and time_control not in criteria.allowed_time_controls:
        return False

    try:
        base_sec, inc_sec = parse_time_control(time_control)
    except ValueError:
        return False

    if criteria.min_base_seconds is not None and base_sec < criteria.min_base_seconds:
        return False

    if criteria.max_base_seconds is not None and base_sec > criteria.max_base_seconds:
        return False

    if criteria.max_estimated_seconds is not None and (base_sec + 40 * inc_sec) >= criteria.max_estimated_seconds:
        return False

    white_elo = _get_elo(game, "WhiteElo")
    black_elo = _get_elo(game, "BlackElo")

    if criteria.min_elo is not None:
        if white_elo is None or white_elo < criteria.min_elo:
            return False
        if black_elo is None or black_elo < criteria.min_elo:
            return False

    if criteria.max_elo is not None:
        if white_elo is not None and white_elo > criteria.max_elo:
            return False
        if black_elo is not None and black_elo > criteria.max_elo:
            return False

    if criteria.max_elo_diff is not None:
        if white_elo is not None and black_elo is not None:
            if abs(white_elo - black_elo) > criteria.max_elo_diff:
                return False

    if criteria.allowed_results:
        result = game.headers.get("Result", "").strip()
        if result not in criteria.allowed_results:
            return False

    if criteria.exclude_terminations:
        termination = game.headers.get("Termination", "").strip()
        if termination in criteria.exclude_terminations:
            return False

    full_moves = game.end().board().fullmove_number
    if criteria.min_full_moves is not None and full_moves < criteria.min_full_moves:
        return False
    if criteria.max_full_moves is not None and full_moves > criteria.max_full_moves:
        return False

    return True


def filter_pgn(
    input_path: Path,
    output_path: Path,
    criteria: Optional[FilterCriteria] = None,
    max_games_to_accept: Optional[int] = None,
) -> GameFilterStats:
    """
    Filters raw PGN games into filtered_games.pgn and returns filter statistics.
    """
    criteria = criteria or FilterCriteria()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    read_count = 0
    accepted_count = 0
    rejected_count = 0
    invalid_count = 0

    with open_pgn_file(input_path) as source, output_path.open("w", encoding="utf-8") as dest:
        for game in read_games(source):
            read_count += 1
            is_valid, _ = validate_game(game)
            if not is_valid:
                invalid_count += 1
            elif game_matches(game, criteria):
                accepted_count += 1
                print(game, file=dest, end="\n\n")
                if max_games_to_accept and accepted_count >= max_games_to_accept:
                    break
            else:
                rejected_count += 1

    return GameFilterStats(
        read=read_count,
        accepted=accepted_count,
        rejected=rejected_count,
        invalid=invalid_count,
    )


def depersonalize_game(game: chess.pgn.Game, game_id: Optional[str] = None) -> chess.pgn.Game:
    """
    Anonymizes player identities to ensure research ethics and data privacy compliance:
    - Replaces White and Black player names with pseudonymous hashes.
    - Sanitizes Site header to base URL + game ID.
    - Preserves remaining clock [%clk ...] comments on moves, while stripping server evals and chat.
    - Preserves Elo ratings, TimeControl, Date, Result, and move legality.
    """
    import hashlib
    import re
    w = game.headers.get("White", "White")
    b = game.headers.get("Black", "Black")
    w_hash = hashlib.sha256(w.encode("utf-8")).hexdigest()[:8]
    b_hash = hashlib.sha256(b.encode("utf-8")).hexdigest()[:8]
    game.headers["White"] = f"Anon_W_{w_hash}"
    game.headers["Black"] = f"Anon_B_{b_hash}"

    site = game.headers.get("Site", "")
    gid = game_id or (site.rstrip("/").split("/")[-1] if "/" in site else "unknown")
    game.headers["Site"] = f"https://lichess.org/{gid}"

    # Retain ONLY clock [%clk hh:mm:ss] comments on moves; strip server %eval and user chat
    clk_pattern = re.compile(r"\[%clk\s+([0-9:]+)\]")
    for node in game.mainline():
        if node.comment:
            match = clk_pattern.search(node.comment)
            node.comment = f"[%clk {match.group(1)}]" if match else ""
        else:
            node.comment = ""

    return game


def stream_filter_and_depersonalize_from_url(
    month: str = "2026-09",
    output_path: Path = Path("data/raw/filtered_games.pgn"),
    criteria: Optional[FilterCriteria] = None,
    target_games: int = 500,
    max_scan_games: int = 25000,
    timeout: int = 15,
) -> GameFilterStats:
    """
    Streams Lichess .zst archive directly from database.lichess.org,
    filters valid rapid human games, depersonalizes them on the fly,
    and saves up to target_games into output_path.
    """
    import io
    import requests
    import zstandard
    from chess_error_prediction.data.download import lichess_month_url

    criteria = criteria or FilterCriteria()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = lichess_month_url(month)
    read_count = 0
    accepted_count = 0
    rejected_count = 0
    invalid_count = 0

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        dctx = zstandard.ZstdDecompressor()
        stream_reader = dctx.stream_reader(response.raw)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")

        with output_path.open("w", encoding="utf-8") as dest:
            while read_count < max_scan_games and accepted_count < target_games:
                try:
                    game = chess.pgn.read_game(text_stream)
                except Exception:
                    invalid_count += 1
                    continue

                if game is None:
                    break

                read_count += 1
                is_valid, _ = validate_game(game)
                if not is_valid:
                    invalid_count += 1
                elif game_matches(game, criteria):
                    accepted_count += 1
                    depersonalized = depersonalize_game(game)
                    print(depersonalized, file=dest, end="\n\n")
                else:
                    rejected_count += 1

    return GameFilterStats(
        read=read_count,
        accepted=accepted_count,
        rejected=rejected_count,
        invalid=invalid_count,
    )
