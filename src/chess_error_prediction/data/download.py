"""
Streaming download utility for Lichess Open Database PGN files.
"""

from pathlib import Path
from typing import Optional
import requests


LICHESS_OPEN_DATABASE_URL = "https://database.lichess.org/standard/lichess_db_standard_{month}.pgn.zst"


def lichess_month_url(month: str) -> str:
    """Return the standard-game Lichess Open Database URL for ``YYYY-MM``."""
    if len(month) != 7 or month[4] != "-" or not month.replace("-", "").isdigit():
        raise ValueError("month must use the YYYY-MM format.")
    return LICHESS_OPEN_DATABASE_URL.format(month=month)


def download_pgn(
    url: str,
    destination: Path,
    chunk_size: int = 1024 * 1024,
    timeout: tuple = (10, 120)
) -> Path:
    """
    Downloads PGN archive (.pgn, .zst, or .gz) from url into destination with streaming chunks.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        output.write(chunk)
    except (requests.RequestException, OSError) as exc:
        raise RuntimeError(f"Unable to download PGN from '{url}': {exc}") from exc

    return destination
