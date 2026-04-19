# Project 2: Interpretability and Explainability


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
