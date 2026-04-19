"""Repo-wide paths shared across parts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path.home() / "ml4h_data" / "p2" / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
