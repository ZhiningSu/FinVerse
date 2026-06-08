# FinWorldModel

**언어:** [English](README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | 한국어

FinWorldModel은 **HMSC** 를 중심으로 한 금융 월드 모델 실험 프레임워크입니다. HMSC는 **Hierarchical Market State Constructor** 의 약자로, 멀티모달 금융 관측을 계층적 시장 상태로 변환하는 것을 목표로 합니다.

## 핵심 아이디어

이 프로젝트는 금융 에이전트가 거래를 실행하기 전에 시장을 내부적으로 어떻게 이해해야 하는지를 연구합니다. 시장 데이터를 단순한 평면 시계열로 보지 않고, HMSC는 다음 정보를 바탕으로 구조화된 시장 상태를 구성합니다.

- OHLCV 및 기술 지표
- 자산 간 구조와 섹터 구조
- 거시 변수와 시장 위험 변수
- 공개 금융 이벤트와 뉴스 신호
- 이산화된 시공간 시장 token

## 모델 구성

- **Dual VQ Market Tokenizer:** 시간적 가격 패턴과 횡단면 시장 구조를 이산 token으로 변환합니다.
- **Multimodal Encoder:** 가격, 뉴스/이벤트, 거시, 그래프 특징을 융합합니다.
- **World Model:** 잠재 시장 상태를 학습하고 미래 상태를 재귀적으로 rollout 합니다.
- **Baselines:** price-only, multimodal no-rollout, no-graph 등 비교 모델을 포함합니다.

## 데이터

현재 목표 데이터 기간은 다음과 같습니다.

```text
2020-01-01 to 2025-12-31
```

주요 데이터 소스:

- Yahoo Finance OHLCV 데이터
- VIX, 달러 지수, 원유, 금, 10년 금리 proxy 등 시장 proxy 변수
- Nasdaq IPO / 공모 이벤트 데이터
- 선택적 SEC EDGAR 및 공개 뉴스/이벤트 수집 스크립트

티커 유니버스:

```text
data/tickers/hmsc_us_90.csv
```

## 가벼운 학습 테스트

디스크 공간이 부족한 경우 `--no-save` 를 사용하세요.

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

## 프로젝트 구조

```text
configs/      학습 설정
datasets/     데이터셋 로더와 배치 처리
models/       월드 모델, 베이스라인, tokenizer
scripts/      데이터 수집 및 전처리 스크립트
trainers/     학습 루프와 손실 함수
train.py      메인 학습 진입점
evaluate.py   평가 진입점
```

## 참고

이 프로젝트는 연구 개발 중입니다. 현재는 HMSC, Dual VQ tokenization, 재귀적 월드 모델 학습 경로를 검증하는 데 초점을 맞추고 있습니다.
