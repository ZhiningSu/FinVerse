<h1 align="center">FinWorldModel</h1>

<p align="center">
  <b>面向金融世界模型的分层市场状态构造器</b>
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
  <a href="README.md">English</a> | 中文 | <a href="README.ja.md">日本語</a> | <a href="README.ko.md">한국어</a>
</p>

<p align="center">
  <b>HMSC</b><br>
  <sub>Hierarchical Market State Constructor</sub>
</p>

> FinWorldModel 是一个围绕 **HMSC** 构建的金融世界模型实验框架。它把多模态金融观测转化为可预测、可想象、可用于策略评估的分层市场状态。

## 核心思想

这个项目研究金融 agent 在执行交易之前，应该如何在内部理解市场。我们不把金融数据简单看成一条扁平时间序列，而是让 HMSC 从以下信息中构造结构化市场状态：

- OHLCV 与技术指标
- 跨资产和行业结构
- 宏观与市场风险变量
- 公开财经事件和新闻信号
- 离散化的时空市场 token

## 模型组件

- **Dual VQ Market Tokenizer:** 将个股时间模式和横截面市场结构转化为离散 token。
- **Multimodal Encoder:** 融合价格、新闻/事件、宏观和图结构特征。
- **World Model:** 学习潜在市场状态，并递推想象未来状态。
- **Baselines:** 包括 price-only、多模态无 rollout、无图结构等对照模型。

## 数据

当前目标数据窗口是：

```text
2020-01-01 到 2025-12-31
```

主要数据来源包括：

- Yahoo Finance OHLCV 数据
- Yahoo 市场代理变量，例如 VIX、美元指数、原油、黄金、10Y 收益率代理
- Nasdaq IPO / 公开发行事件数据
- 可选的 SEC EDGAR 与公开新闻/事件采集脚本

股票池文件位于：

```text
data/tickers/hmsc_us_90.csv
```

## 运行轻量训练测试

磁盘空间不足时建议使用 `--no-save`：

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

## 项目结构

```text
configs/      训练配置
datasets/     数据集读取与 batch 构造
models/       世界模型、基线模型和 tokenizer
scripts/      数据采集与预处理脚本
trainers/     训练循环和损失函数
train.py      主训练入口
evaluate.py   评估入口
```

## 说明

本项目仍处于研究开发阶段。当前重点是验证 HMSC、Dual VQ tokenization 和递推式世界模型训练链路，然后再扩展到更大规模实验。
