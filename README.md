# Project 2: Interpretability and Explainability

## Project structure

```
ml4hc-project2/
├── common/          # helpers shared across parts (metrics, plotting, utils)
├── part1/           # heart disease
├── part2/           # pneumonia
├── outputs/         # generated plots and results
│   ├── part1/
│   └── part2/
├── requirements.txt
└── README.md
```

## Data

Data is expected at `~/ml4h_data/p2/data/` and is not checked into the repository. The directory should contain:

```
~/ml4h_data/p2/data/
├── heart.csv              # data for part 1 on heart disease
└── chest_xray/            # data for part 2 on pneumonia
    ├── train/
    │   ├── NORMAL/        # images, no pneumonia
    │   └── PNEUMONIA/     # images, pneumonia
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

## Setup

We assume Python v3.12.3 (which should be the default on the cluster).

```bash
cd ml4hc-project1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Part 2

### Model

The Convolutional Network model is implemented in model.py and contains two implementations:
    - resnet: A ResNet inspired network, which uses skip-connections. The current implementation has 8 blocks, where each block has two 3x3 convolutional layers,
a ReLU activation, BatchNorm and there is a skip connection from the beginning of the block to the final relu.
    - densenet: A DenseNet spired network, which has 4 blocks, and each block is connected to all the subsequent blocks, where the inputs are concatenated instead
of added.

In order to train and evaluate the model on the cluster, you can use the following:
```bash
sbatch run_pipeline.sh
```

If you'd like to simply train the model, you can use:
```bash
pip instrall -r requirements.txt
TRAIN_PATH="$HOME/ml4h_data/p2/data/chest_xray/train"
TEST_PATH="$HOME/ml4h_data/p2/data/chest_xray/test"
MODEL_WEIGHTS="output/part2/models/densenet.pth"
OUTPUT_DIR="outputs/part2/normal"

mkdir -p outputs/part2/models
mkdir -p outputs/part2/normal

pip install -r requirements.txt

PYTHONUNBUFFERED=1 python3 -m part2.train \
  --train_path "$TRAIN_PATH" \
  --model_type="densenet" \
  --output_model "$MODEL_WEIGHTS" \
  --batch_size 32 \
  --epochs 20
```

We also offer the ability to train the shuffled model (as per Q5) on the cluster:
```bash
sbatch run_pipeline_shuffled.sh
```

Or manually:
```bash
PYTHONUNBUFFERED=1 python3 -m part2.train \
  --train_path "$TRAIN_PATH" \
  --model_type="densenet" \
  --output_model "$MODEL_WEIGHTS" \
  --batch_size 32 \
  --shuffled \
  --epochs 20
```

### Integrated gradients and Grad-CAM

They are implemented correspondingly in ig.py and gradcam.py. The two techniques try to create heatmaps of the norm of the most imporant gradients via different methodologies:
    - Integrated Gradients: Trying to compute a gradient along the entire path from the output to the input and highlight their importance.
    - Grad-CAM: Trying to create an interpolated activation map from the final layer onto the input image.

The pipeline already generated the plots:
```bash
sbatch run_pipeline.sh
sbatch run_pipeline_shuffled.sh # Optionally for the shuffled model
```

You can manually use the plot_grid.py script to generate both Grad-CAM activation maps as well as a visualization of Integrated Gradients:
```bash
PYTHONUNBUFFERED=1 python3 -m part2.plot_grid \
  --train_path "$TRAIN_PATH" \
  --test_path "$TEST_PATH" \
  --model_type "densenet" \
  --model_weights "$MODEL_WEIGHTS" \
  --output_dir "$OUTPUT_DIR"
```