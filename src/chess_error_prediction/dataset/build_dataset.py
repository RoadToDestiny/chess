"""
Dataset construction pipeline from parsed PGN games and Stockfish evaluations.
Extracts position records, extracts pre-move features, and computes post-move error labels.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional
import chess
import chess.pgn
import numpy as np
import pandas as pd

from chess_error_prediction.features.extractors import extract_all_features
from chess_error_prediction.labeling.error_labeler import (
    compute_centipawn_loss,
    assign_blunder_label,
)
from chess_error_prediction.schemas import PositionRecord
from chess_error_prediction.data.pgn_reader import open_pgn_file, read_games, parse_time_control


def build_dataset_from_game(
    game: chess.pgn.Game,
    evaluator: Callable[[chess.Board], Dict[str, Any]],
    game_id: Optional[str] = None,
    threshold_cp: float = 200.0,
) -> List[PositionRecord]:
    """
    Extracts complete position records for every ply in a single chess game.
    Evaluator takes a chess.Board and returns dict {"score_cp": float, "is_mate": bool, ...}.
    """
    gid = game_id or game.headers.get("Site") or game.headers.get("Event", "game_001")

    # Player ratings
    white_elo_raw = game.headers.get("WhiteElo", "1500")
    black_elo_raw = game.headers.get("BlackElo", "1500")
    white_elo = float(white_elo_raw) if str(white_elo_raw).isdigit() else 1500.0
    black_elo = float(black_elo_raw) if str(black_elo_raw).isdigit() else 1500.0

    # Time control
    time_control_str = game.headers.get("TimeControl", "600+0")
    try:
        base_time, inc_time = parse_time_control(time_control_str)
    except (ValueError, AttributeError):
        base_time, inc_time = 600, 0

    board = game.board()
    records: List[PositionRecord] = []
    prev_eval_white: float = 0.0

    for ply_idx, move in enumerate(game.mainline_moves(), start=1):
        move_number = (ply_idx + 1) // 2
        is_white = (board.turn == chess.WHITE)
        color_str = "w" if is_white else "b"

        # 1. Evaluate board BEFORE move (Stockfish returns White's POV)
        eval_before = evaluator(board)
        score_before_white = float(eval_before.get("score_cp", 0.0) or 0.0)
        is_mate_before = bool(eval_before.get("is_mate", False))

        # Position eval from current player's POV
        eval_before_player = score_before_white if is_white else -score_before_white
        eval_diff_prev_move = score_before_white - prev_eval_white

        # 2. Extract features before move
        features = extract_all_features(
            board=board,
            eval_before_player_pov=eval_before_player,
            eval_diff_prev_move=eval_diff_prev_move,
            white_elo=white_elo,
            black_elo=black_elo,
            time_control_base=float(base_time),
            time_control_inc=float(inc_time),
        )

        # 3. Execute move
        board.push(move)

        # 4. Evaluate board AFTER move
        eval_after = evaluator(board)
        score_after_white = float(eval_after.get("score_cp", 0.0) or 0.0)
        is_mate_after = bool(eval_after.get("is_mate", False))

        eval_after_player = score_after_white if is_white else -score_after_white
        is_mate_eval = is_mate_before or is_mate_after

        # 5. Compute loss and label blunder
        if is_mate_eval:
            eval_loss = 0.0
            is_error = 0
        else:
            eval_loss = compute_centipawn_loss(score_before_white, score_after_white, color=color_str)
            is_error = assign_blunder_label(eval_loss, threshold_cp=threshold_cp)

        record = PositionRecord(
            game_id=str(gid),
            ply=ply_idx,
            move_number=move_number,
            color=color_str,
            eval_before=round(eval_before_player, 2),
            eval_diff_prev_move=round(eval_diff_prev_move, 2),
            material_balance=features["material_balance"],
            piece_count=features["piece_count"],
            pawn_count=features["pawn_count"],
            legal_checks_count=features["legal_checks_count"],
            legal_captures_count=features["legal_captures_count"],
            attacked_pieces_count=features["attacked_pieces_count"],
            game_phase=features["game_phase"],
            player_rating=features["player_rating"],
            opponent_rating=features["opponent_rating"],
            rating_diff=features["rating_diff"],
            time_control_base=features["time_control_base"],
            time_control_inc=features["time_control_inc"],
            eval_after=round(eval_after_player, 2),
            eval_loss=round(eval_loss, 2),
            is_error=is_error,
            is_mate_eval=is_mate_eval,
            fen=board.fen(),
        )
        records.append(record)
        prev_eval_white = score_after_white

    return records


def build_dataset_from_pgn(
    pgn_path: Path,
    evaluator: Callable[[chess.Board], Dict[str, Any]],
    max_games: Optional[int] = None,
    threshold_cp: float = 200.0,
) -> pd.DataFrame:
    """
    Parses a PGN file, extracts features, analyzes with evaluator, and builds a DataFrame.
    """
    pgn_path = Path(pgn_path)
    records: List[Dict[str, Any]] = []

    with open_pgn_file(pgn_path) as handle:
        for game_idx, game in enumerate(read_games(handle), start=1):
            gid = game.headers.get("Site") or f"pgn_game_{game_idx:04d}"
            game_records = build_dataset_from_game(
                game=game,
                evaluator=evaluator,
                game_id=gid,
                threshold_cp=threshold_cp,
            )
            records.extend([r.to_dict() for r in game_records])
            if max_games and game_idx >= max_games:
                break

    return pd.DataFrame(records)


def generate_synthetic_dataset(
    num_games: int = 50,
    moves_per_game_range: tuple = (25, 45),
    error_threshold_cp: float = 200.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a deterministic synthetic chess dataset adhering strictly to the schema.
    Used for end-to-end tests and dry runs without requiring full Stockfish analysis.
    """
    rng = np.random.default_rng(random_seed)
    records = []

    for game_idx in range(1, num_games + 1):
        game_id = f"lichess_game_{game_idx:04d}"
        num_moves = rng.integers(moves_per_game_range[0], moves_per_game_range[1] + 1)
        total_plies = num_moves * 2

        white_rating = float(rng.integers(1400, 2400))
        black_rating = float(rng.integers(1400, 2400))
        base_time = 600.0
        inc_time = float(rng.choice([0.0, 5.0, 10.0]))

        current_eval_white = 0.0
        prev_eval_white = 0.0

        for ply in range(1, total_plies + 1):
            move_number = (ply + 1) // 2
            is_white = (ply % 2 != 0)
            color = "w" if is_white else "b"

            player_rating = white_rating if is_white else black_rating
            opponent_rating = black_rating if is_white else white_rating
            rating_diff = player_rating - opponent_rating

            if move_number <= 10:
                game_phase = 1
            elif move_number <= 30:
                game_phase = 2
            else:
                game_phase = 3

            pawn_count = max(0, min(16, 16 - int((ply / total_plies) * 10) + rng.integers(-1, 2)))
            piece_count = max(2, min(32, 32 - int((ply / total_plies) * 20) + rng.integers(-2, 2)))
            material_balance = float(rng.integers(-6, 7))
            legal_checks_count = int(rng.choice([0, 1, 2], p=[0.75, 0.20, 0.05]))
            legal_captures_count = int(rng.integers(0, 6))
            attacked_pieces_count = int(rng.integers(0, 5))

            eval_before_player = current_eval_white if is_white else -current_eval_white
            eval_diff_prev_move = current_eval_white - prev_eval_white

            # Error probability increases with tactical complexity and lower rating
            tactical_stress = (attacked_pieces_count * 0.03) + (legal_checks_count * 0.02)
            error_prob = 0.07 + tactical_stress - (player_rating - 1500) * 0.00004
            error_prob = float(np.clip(error_prob, 0.03, 0.28))

            is_error = 1 if rng.random() < error_prob else 0

            if is_error:
                eval_loss = float(rng.uniform(error_threshold_cp, 650.0))
            else:
                eval_loss = float(rng.uniform(0.0, error_threshold_cp - 15.0))

            if is_white:
                next_eval_white = current_eval_white - eval_loss
                eval_after_player = eval_before_player - eval_loss
            else:
                next_eval_white = current_eval_white + eval_loss
                eval_after_player = eval_before_player - eval_loss

            prev_eval_white = current_eval_white
            current_eval_white = float(np.clip(next_eval_white, -2500.0, 2500.0))

            records.append({
                "game_id": game_id,
                "ply": ply,
                "move_number": move_number,
                "color": color,
                "eval_before": round(eval_before_player, 1),
                "eval_diff_prev_move": round(eval_diff_prev_move, 1),
                "material_balance": material_balance,
                "piece_count": piece_count,
                "pawn_count": pawn_count,
                "legal_checks_count": legal_checks_count,
                "legal_captures_count": legal_captures_count,
                "attacked_pieces_count": attacked_pieces_count,
                "game_phase": game_phase,
                "player_rating": player_rating,
                "opponent_rating": opponent_rating,
                "rating_diff": rating_diff,
                "time_control_base": base_time,
                "time_control_inc": inc_time,
                "eval_after": round(eval_after_player, 1),
                "eval_loss": round(eval_loss, 1),
                "is_error": is_error,
                "is_mate_eval": False,
            })

    return pd.DataFrame(records)
