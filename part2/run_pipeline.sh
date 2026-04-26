#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=ml4h_jobs
#SBATCH --output=run_pipeline_%j.out
#SBATCH --mem=16G

module load cuda/12.6

cd ~/ml4hc-project2

VENV_DIR="$HOME/ml4hc_env"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

TRAIN_DATA="$HOME/ml4h_data/p2/data/chest_xray/train"
TEST_DATA="$HOME/ml4h_data/p2/data/chest_xray/test"
MODEL_WEIGHTS="output/part2/models/densenet.pth"
OUTPUT_DIR="outputs/part2/normal"

mkdir -p outputs/part2/models
mkdir -p outputs/part2/normal

pip install -r requirements.txt

PYTHONUNBUFFERED=1 python3 -m part2.train \
  --train_path "$TRAIN_PATH" \
  --model_type="densenet" \
  --output_model "$MODEL_WEIGHTS" \
  --epochs 20

PYTHONUNBUFFERED=1 python3 -m part2.evaluate \
  --train_path "$TRAIN_PATH" \
  --test_path "$TEST_PATH" \
  --model_type "densenet" \
  --model_weights "$MODEL_WEIGHTS" \
  --output_dir "$OUTPUT_DIR"

PYTHONUNBUFFERED=1 python3 -m part2.plot_grid \
  --train_path "$TRAIN_PATH" \
  --test_path "$TEST_PATH" \
  --model_type "densenet" \
  --model_weights "$MODEL_WEIGHTS" \
  --output_dir "$OUTPUT_DIR"
