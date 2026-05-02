"""Configuration for Part 1 (heart disease)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path.home() / "ml4h_data" / "p2" / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DATA_PATH = DATA_DIR / "heart.csv"
SPLITS_PATH = DATA_DIR / "splits_heart.csv"
OUTPUT_DIR = OUTPUT_ROOT / "part1"

LABEL = "HeartDisease"
NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
BINARY_FEATURES = ["FastingBS"]
CATEGORICAL_FEATURES = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
ORDINAL_FEATURES = {
    "ST_Slope": ["Up", "Flat", "Down"],  # Up=0 (normal response) → Down=2 (highest risk)
}
