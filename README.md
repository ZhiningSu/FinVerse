# FinWorldModel

**Languages:** English | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

FinWorldModel is an experimental framework for building a financial world model around **HMSC**: the **Hierarchical Market State Constructor**. HMSC converts multimodal financial observations into hierarchical market states that can support forecasting, imagination, and strategy evaluation.

## Core Idea

The project studies how a financial agent can internally understand the market before execution. Instead of treating market data as a flat time series, HMSC constructs structured state representations from:

- OHLCV and technical market features
- cross-asset and sector structure
- macro and market-risk variables
- public financial events and news signals
- discrete spatio-temporal market tokens

## Model Components

- **Dual VQ Market Tokenizer:** converts temporal price patterns and cross-sectional market structure into discrete tokens.
- **Multimodal Encoder:** fuses price, news/event, macro, and graph features.
- **World Model:** learns latent market states and performs recurrent future rollout.
- **Baselines:** price-only, multimodal without rollout, and no-graph variants.

## Data

The current target data window is:

```text
2020-01-01 to 2025-12-31
```

Main data sources include:

- Yahoo Finance OHLCV data
- Yahoo market proxy variables such as VIX, dollar index, crude oil, gold, and 10Y yield proxy
- Nasdaq IPO/public offering event data
- optional SEC EDGAR and public news/event collection scripts

The ticker universe is stored in:

```text
data/tickers/hmsc_us_90.csv
```

## Run A Smoke Training Test

Use `--no-save` when disk space is limited:

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

## Project Structure

```text
configs/      training configuration
datasets/     dataset loader and batching logic
models/       world model, baselines, and tokenizers
scripts/      data collection and preprocessing scripts
trainers/     training loop and loss functions
train.py      main training entry
evaluate.py   evaluation entry
```

## Notes

This project is under active research development. The current implementation focuses on validating HMSC, Dual VQ tokenization, and recurrent world-model learning before scaling to larger experiments.
