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
python ml_pipeline.py                # Train & evaluate scikit-learn models
streamlit run dashboard_app.py       # Launch interactive prediction UI

# PyTorch alternative
python ml_pipeline_pytorch.py        # Train PyTorch models (dashboard still uses joblib)

# Research workflow
python run_experiments.py            # Sweep RANDOMNESS 0.1–0.9, collect metrics
python visualization_analysis.py     # Generate comparison plots
```

No test suite exists; evaluation is done via printed metrics, confusion matrices, and exported predictions.

## Architecture

Five independent scripts share data through the filesystem:

```
dataset_generator.py  →  data/hotel_reviews_synthetic.csv
                              ↓
ml_pipeline.py        →  models/*.joblib + data/predictions_*.csv + plots/
ml_pipeline_pytorch.py →  models/*.pth   + data/predictions_pytorch_*.csv
                              ↓
dashboard_app.py      ←  models/*.joblib (loads with @st.cache_resource)
```

- **run_experiments.py** orchestrates dataset_generator + ml_pipeline across multiple RANDOMNESS values, saving JSON results to `experiments/`.
- **visualization_analysis.py** produces comparative plots (accuracy, F1 heatmap, paradox visualization).

## Key Design Decisions

- **RANDOMNESS parameter** (0.0–1.0) in `dataset_generator.py` controls label noise injection — the central variable of the research.
- **Dual-task, dual-framework**: Separate models for department and sentiment, implemented in both scikit-learn and PyTorch for comparison.
- **Pipeline pattern**: scikit-learn models wrap TF-IDF + LogisticRegression in a `Pipeline` object, serialized with joblib.
- **Reproducibility**: `random_state=42` used throughout.
- **Dashboard loads only joblib models** — PyTorch models are for experimental comparison only.
- **Preprocessing is intentionally minimal** (lowercase, remove punctuation/numbers) — no lemmatization or stemming.

## Dependencies

Managed via `dependencies.txt` (not requirements.txt). Key packages: pandas, numpy, scikit-learn, torch, matplotlib, seaborn, streamlit, altair==4.2.2.
