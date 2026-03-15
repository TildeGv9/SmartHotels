# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmartHotels is an academic ML project that classifies synthetic hotel reviews by **department** (Housekeeping, Reception, F&B) and **sentiment** (pos, neg, neu). The core research contribution is the "Randomness Paradox": sentiment analysis degrades gracefully under label noise while department classification collapses.

Documentation and comments are in Italian (academic requirement).

## Common Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r dependencies.txt

# Standard workflow
python dataset_generator.py          # Generate synthetic dataset (450 reviews)
python ml_pipeline_sklearn.py        # Train & evaluate scikit-learn models
streamlit run dashboard_app.py       # Launch interactive prediction UI

# PyTorch alternative
python ml_pipeline_pytorch.py        # Train PyTorch models (dashboard still uses joblib)

# Research workflow
python run_experiments.py                    # Sweep RANDOMNESS 0.1–0.9 (default: pytorch)
python run_experiments.py --pipeline sklearn  # Use scikit-learn pipeline
python run_experiments.py --pipeline pytorch  # Use PyTorch pipeline
python visualization_analysis.py             # Generate comparison plots
```

No test suite exists; evaluation is done via printed metrics, confusion matrices, and exported predictions.

## Architecture

Five independent scripts share data through the filesystem:

```
dataset_generator.py  →  data/hotel_reviews_synthetic_{RANDOMNESS}.csv (temporaneo)
                              ↓
run_experiments.py    →  sposta in data/{sklearn|pytorch}/reviews/
                              ↓
ml_pipeline_sklearn.py →  models/*.joblib + data/sklearn/predictions/predictions_..._R{R}_sklearn.csv + plots/sklearn/
ml_pipeline_pytorch.py →  models/*.pth   + data/pytorch/predictions/predictions_..._R{R}_pytorch.csv + plots/pytorch/
                              ↓
dashboard_app.py      ←  models/*.joblib OR models/*.pth (selezionabile da sidebar)
```

### Struttura directory `data/`
```
data/
├── sklearn/
│   ├── reviews/
│   │   └── hotel_reviews_synthetic_{R}.csv    # spostato da run_experiments.py
│   └── predictions/
│       └── predictions_YYYYMMDD_HHMMSS_R{R}_sklearn.csv
└── pytorch/
    ├── reviews/
    │   └── hotel_reviews_synthetic_{R}.csv    # spostato da run_experiments.py
    └── predictions/
        └── predictions_YYYYMMDD_HHMMSS_R{R}_pytorch.csv
```

### Struttura directory `plots/`
```
plots/
├── sklearn/
│   ├── accuracy_comparison_sklearn.png
│   ├── paradox_visualization_sklearn.png
│   ├── f1_heatmap_sklearn.png
│   ├── dashboard_complete_sklearn.png
│   └── current_performance_sklearn.png
└── pytorch/
    ├── accuracy_comparison_pytorch.png
    ├── paradox_visualization_pytorch.png
    ├── f1_heatmap_pytorch.png
    ├── dashboard_complete_pytorch.png
    └── current_performance_pytorch.png
```

- **run_experiments.py** orchestrates dataset_generator + ml_pipeline across multiple RANDOMNESS values, saving JSON results to `experiments/`. Generates datasets in `data/`, then moves them into `data/{pipeline}/reviews/` (the standalone file in `data/` is deleted after the copy). Passes `DATASET_PATH`/`PREDICTIONS_DIR` env vars to pipelines. Supports both pipelines via `--pipeline sklearn|pytorch`.
- **visualization_analysis.py** produces comparative plots (accuracy, F1 heatmap, paradox visualization) in `plots/{pipeline_name}/` when a pipeline name is set.

## Key Design Decisions

- **RANDOMNESS parameter** (0.0–1.0) in `dataset_generator.py` controls label noise injection — the central variable of the research.
- **Dual-task, dual-framework**: Separate models for department and sentiment, implemented in both scikit-learn and PyTorch for comparison.
- **Pipeline pattern**: scikit-learn models wrap TF-IDF + LogisticRegression in a `Pipeline` object, serialized with joblib.
- **Reproducibility**: `random_state=42` used throughout.
- **Dashboard supports both pipelines** — sidebar radio button lets the user switch between scikit-learn (joblib) and PyTorch models at runtime.
- **Preprocessing is intentionally minimal** (lowercase, remove punctuation/numbers) — no lemmatization or stemming.
- **Env var overrides**: `ml_pipeline_sklearn.py` and `ml_pipeline_pytorch.py` read `DATASET_PATH` and `PREDICTIONS_DIR` from environment variables when set (used by `run_experiments.py`), falling back to `data/{sklearn|pytorch}/reviews/` and `data/{sklearn|pytorch}/predictions/` defaults.
- **RANDOMNESS in filenames**: Dataset files include the RANDOMNESS value as suffix (e.g., `hotel_reviews_synthetic_0.9.csv`). Pipelines extract this value from the filename to include it in prediction output names (`_R0.9_`).

## Best Practices

- **Document every change**: When modifying code (renaming files, changing output paths, adding parameters, etc.), always update all related documentation (CLAUDE.md, README.md) and code references (comments, error messages) to reflect the change. No undocumented changes.
- **Pipeline-specific output naming**: All output files (models, predictions, plots, experiment JSON) must include a `_sklearn` or `_pytorch` suffix to distinguish which pipeline generated them and avoid overwriting.

## Dependencies

Managed via `dependencies.txt` (not requirements.txt). Key packages: pandas, numpy, scikit-learn, torch, matplotlib, seaborn, streamlit, altair==4.2.2.
