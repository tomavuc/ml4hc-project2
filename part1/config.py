"""Configuration for Part 1 (heart disease)."""

from common.config import DATA_DIR, OUTPUT_ROOT

DATA_PATH = DATA_DIR / "heart.csv"
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
