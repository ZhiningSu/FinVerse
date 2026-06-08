# FinWorldModel

**言語:** [English](README.md) | [中文](README.zh-CN.md) | 日本語 | [한국어](README.ko.md)

FinWorldModel は、**HMSC** を中心とした金融世界モデルの実験フレームワークです。HMSC は **Hierarchical Market State Constructor** の略で、多モーダルな金融観測を階層的な市場状態へ変換することを目的としています。

## 基本アイデア

本プロジェクトは、金融エージェントが取引を実行する前に、どのように市場を内部的に理解すべきかを研究します。市場データを単なる平坦な時系列として扱うのではなく、HMSC は以下の情報から構造化された市場状態を構築します。

- OHLCV とテクニカル指標
- 資産間およびセクター構造
- マクロ変数と市場リスク変数
- 公開金融イベントとニュース信号
- 離散化された時空間市場 token

## モデル構成

- **Dual VQ Market Tokenizer:** 時系列価格パターンと横断的な市場構造を離散 token に変換します。
- **Multimodal Encoder:** 価格、ニュース/イベント、マクロ、グラフ特徴を融合します。
- **World Model:** 潜在市場状態を学習し、未来状態を再帰的に rollout します。
- **Baselines:** price-only、多モーダル no-rollout、no-graph などの比較モデルを含みます。

## データ

現在の対象期間は以下です。

```text
2020-01-01 から 2025-12-31
```

主なデータソース:

- Yahoo Finance の OHLCV データ
- VIX、ドル指数、原油、金、10年金利 proxy などの市場 proxy 変数
- Nasdaq IPO / 公募イベントデータ
- SEC EDGAR と公開ニュース/イベント収集スクリプト

銘柄ユニバース:

```text
data/tickers/hmsc_us_90.csv
```

## 軽量トレーニングテスト

ディスク容量が少ない場合は `--no-save` を使用します。

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

## プロジェクト構成

```text
configs/      学習設定
datasets/     データセットとバッチ処理
models/       世界モデル、ベースライン、tokenizer
scripts/      データ収集と前処理
trainers/     学習ループと損失関数
train.py      メイン学習入口
evaluate.py   評価入口
```

## 注記

本プロジェクトは研究開発中です。現在は HMSC、Dual VQ tokenization、再帰的世界モデル学習の検証に重点を置いています。
