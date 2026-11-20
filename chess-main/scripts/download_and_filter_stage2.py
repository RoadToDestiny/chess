"""
Stage 2: Download, filter, and depersonalize games from Lichess Open Database (2026-09).
"""

import sys
from pathlib import Path
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from chess_error_prediction.data.game_filter import (
    FilterCriteria,
    stream_filter_and_depersonalize_from_url,
)


def main() -> None:
    output_pgn = ROOT_DIR / "data" / "raw" / "filtered_games.pgn"
    print(f"Starting streaming filter from Lichess Open Database (month: 2026-09)...")
    print(f"Criteria: Rapid (>= 600s), Elo 1600-2800, Moves 10-120, Humans only (no BOTs), Standard variant.")
    print(f"Destination: {output_pgn}")

    start_time = time.time()
    criteria = FilterCriteria(
        min_elo=1600,
        max_elo=2800,
        min_base_seconds=600,
        min_full_moves=10,
        max_full_moves=120,
        exclude_bots=True,
        require_standard_variant=True,
    )

    stats = stream_filter_and_depersonalize_from_url(
        month="2026-09",
        output_path=output_pgn,
        criteria=criteria,
        target_games=500,
        max_scan_games=20000,
        timeout=20,
    )

    elapsed = time.time() - start_time
    file_size_mb = output_pgn.stat().st_size / (1024 * 1024) if output_pgn.exists() else 0

    print("=" * 60)
    print("STAGE 2 ARTIFACT GENERATION COMPLETED:")
    print(f"  Scanned games from stream:  {stats.read}")
    print(f"  Accepted & depersonalized:  {stats.accepted}")
    print(f"  Rejected by criteria:       {stats.rejected}")
    print(f"  Invalid / unparseable:      {stats.invalid}")
    print(f"  File size on disk:          {file_size_mb:.2f} MB")
    print(f"  Elapsed time:               {elapsed:.1f} seconds")
    print(f"  Artifact location:          {output_pgn}")
    print("=" * 60)


if __name__ == "__main__":
    main()
