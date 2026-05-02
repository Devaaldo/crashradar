# CrashRadar

**Stock crash early warning system for IDX equities using XGBoost and SHAP.**

Classifies whether an IDX stock is at risk of dropping more than 10% within the next 5 trading days. Built as a proof of concept for explainable binary classification on financial time series.

---

## Overview

CrashRadar does not predict prices. It assigns a crash probability score to each trading day based on technical indicators derived from OHLCV data. SHAP values make every prediction interpretable at the feature level.

| Property | Value |
|---|---|
| Model | XGBoost Classifier |
| Task | Binary classification (crash / safe) |
| Horizon | 5 trading days |
| Label | Min forward return < -10% |
| Features | 8 OHLCV-derived technical indicators |
| Explainability | SHAP TreeExplainer |
| Default ticker | BBCA.JK (Bank Central Asia) |
| Data window | 10 years (~2,460 trading days) |

---

## Project Structure

```
crashradar/
├── notebooks/
│   └── crashradar.ipynb      # Main analysis notebook
├── outputs/
│   ├── figures/              # Auto-generated plots (git-ignored)
│   ├── models/               # Saved model artifacts (git-ignored)
│   └── data/                 # Processed feature parquet (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/devaaldo/crashradar.git
cd crashradar
pip install -r requirements.txt
jupyter notebook notebooks/crashradar.ipynb
```

Run all cells top to bottom. The notebook fetches data, trains the model, evaluates it, and produces a real-time signal for the latest available trading day.

To analyze a different IDX stock, change `TICKER` in the **Configuration** cell (e.g. `"BBRI.JK"`, `"TLKM.JK"`).

---

## Features

| Feature | Description |
|---|---|
| `rsi_14` | RSI (14-period) — momentum and overbought signal |
| `volume_spike` | Volume / 20-day average — unusual activity detector |
| `bb_position` | Normalised position within Bollinger Bands |
| `macd_diff` | MACD histogram — trend shift signal |
| `return_5d` | 5-day lagged return — recent momentum |
| `atr_14` | ATR normalised by close — volatility regime |
| `price_vs_ma20` | Distance from 20-day MA — mean-reversion signal |
| `high_low_range` | Daily range normalised by close — intraday volatility |

---

## Outputs

Each run produces six figures saved to `outputs/figures/`:

| File | Content |
|---|---|
| `01_crash_labels.png` | Price chart with crash warning days and return distribution |
| `02_feature_correlation.png` | Feature correlation heatmap |
| `03_model_evaluation.png` | Confusion matrix, ROC curve, predicted probability distribution |
| `04_shap_importance.png` | Global feature importance (mean absolute SHAP) |
| `05_shap_beeswarm.png` | SHAP value distribution per feature |
| `06_shap_waterfall_latest.png` | Per-prediction SHAP waterfall for the latest trading day |

Trained model artifacts are saved to `outputs/models/` in both `.json` (XGBoost native) and `.pkl` (joblib) formats.

---

## Evaluation Note

Crash events are rare by definition (~0.6% of trading days for BBCA.JK over 10 years). With a chronological 80/20 split, it is possible that all crash events fall within the training window, leaving the test set with no positive labels. In that case, ROC-AUC and PR-AUC are reported as N/A, and the evaluation panel shows the predicted probability distribution instead of the ROC curve. This is a known limitation of evaluating rare-event classifiers on short out-of-sample periods.

---

## Kernel Restart Recovery

If the kernel restarts mid-session, run only the **Kernel Restart Recovery** cell. It reloads the saved model and feature dataset from `outputs/` and reconstructs the train/test splits without re-downloading or re-training.

---

## Requirements

See [requirements.txt](requirements.txt). Python 3.11+ recommended.

Core dependencies:

- `yfinance` — market data ingestion
- `xgboost` — gradient boosted tree classifier
- `shap` — model explainability
- `scikit-learn` — metrics and evaluation
- `pandas`, `numpy`, `matplotlib`, `seaborn` — data processing and visualization
- `joblib`, `pyarrow` — model and data serialization
- `jupyter`, `notebook`, `ipywidgets` — notebook environment
- `pandas-ta` — optional; used for RSI if available, falls back to manual calculation on Python 3.11+

---

## Disclaimer

This project is for research and educational purposes only. It does not constitute investment advice. Past patterns in historical data do not guarantee future market outcomes.
