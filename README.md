<h1 align="center">FinWorldModel</h1>

<p align="center">
  <b>A Hierarchical Market State Constructor for Financial World Models</b>
</p>

<p align="center">
  <a href="https://github.com/ZhiningSu/FinVerse">
    <img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-FinVerse-24292f?logo=github">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-World%20Model-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="HMSC" src="https://img.shields.io/badge/Core-HMSC-6f42c1">
  <img alt="Status" src="https://img.shields.io/badge/Status-Research%20Prototype-2ea44f">
</p>

<p align="center">
  English | <a href="README.zh-CN.md">中文</a> | <a href="README.ja.md">日本語</a> | <a href="README.ko.md">한국어</a>
</p>

<p align="center">
  <b>HMSC</b><br>
  <sub>Hierarchical Market State Constructor</sub>
</p>

> FinWorldModel builds a financial agent's internal market-state constructor: it transforms multimodal market observations into hierarchical latent states that support forecasting, imagination, and strategy evaluation.

## Why FinWorldModel?

Financial markets are not flat numerical sequences. They are structured systems shaped by assets, sectors, regimes, macro conditions, and public events. FinWorldModel studies how an agent can construct an internal representation of this world before making downstream decisions.

At the center of the project is **HMSC**, a module designed to answer:

```text
What is the current market state, and how may it evolve under different strategies?
```

## Highlights

- **Hierarchical state construction:** asset-level, cross-sectional, event-aware, and market-level representations.
- **Dual VQ tokenization:** discrete temporal and cross-sectional market tokens inspired by VQ-VAE style financial tokenization.
- **Multimodal market modeling:** OHLCV, technical features, macro proxies, public events, and graph structure.
- **Recurrent world model:** latent-state rollout for future market imagination.
- **Lightweight experimentation:** CPU-friendly smoke tests and `--no-save` mode for limited local disk space.

## Architecture

```text
OHLCV / technical features
macro and market-risk variables
public financial events and news signals
cross-asset graph structure
        │
        ▼
HMSC: Hierarchical Market State Constructor
        │
        ├── Temporal VQ Tokens
        ├── Cross-sectional VQ Tokens
        ├── Multimodal State Fusion
        └── Latent Market State z_t
        │
        ▼
Recurrent World Model
        │
        ▼
Forecasting / Imagination / Strategy Evaluation
```

## Data Scope

The current target window is:

```text
2020-01-01 to 2025-12-31
```

The project currently supports:

- Yahoo Finance OHLCV data
- market proxies such as VIX, dollar index, crude oil, gold, and 10Y yield proxy
- Nasdaq IPO and public-offering events
- optional SEC EDGAR and public news/event collection scripts

Ticker universe:

```text
data/tickers/hmsc_us_90.csv
```

## Quick Start

Run a lightweight smoke training test:

```bash
python train.py \
  --data-root data/processed/real \
  --output-dir outputs/dual_vq_trial_nosave \
  --model full \
  --num-epochs 3 \
  --batch-size 16 \
  --max-train-episodes 256 \
  --max-val-episodes 64 \
  --device cpu \
  --vq-weight 0.05 \
  --no-save
```

Use `--no-save` when disk space is limited.

## Repository Layout

```text
configs/      training configuration
datasets/     dataset loader and batching logic
models/       world model, baselines, and tokenizers
scripts/      data collection and preprocessing scripts
trainers/     training loop and loss functions
train.py      main training entry
evaluate.py   evaluation entry
```

## Research Status

FinWorldModel is an active research prototype. The current implementation focuses on validating HMSC, Dual VQ market tokenization, and recurrent world-model learning before scaling to larger multimodal experiments.
