export type StageStatus = "idle" | "running" | "success" | "failed";
export type Market = "us" | "cn";
export type Language = "en" | "zh";

export type PipelineStage = {
  name: string;
  status: StageStatus;
  duration_sec?: number;
  message?: string;
};

export type Strategy = {
  name: string;
  description: string;
  confidence: number;
};

export type RegimeProbs = {
  bear: number;
  sideway: number;
  bull: number;
};

export type MarketState = {
  regime: string;
  regime_probs: RegimeProbs;
  market_return_20d: number;
  market_vol_20d: number;
  latent_summary: {
    pc1: number;
    pc2: number;
    nearest_historical_regime: string;
  };
};

export type AssetRecommendation = {
  rank: number;
  ticker: string;
  name: string;
  type: "stock" | "etf";
  sector: string;
  close: number;
  score: number;
  expected_return_30d: number;
  predicted_volatility: number;
  predicted_downside: number;
  regime_probs: RegimeProbs;
  reasons: string[];
};

export type IndustryRecommendation = {
  rank: number;
  sector: string;
  score: number;
  news_score: number;
  news_count: number;
  macro_score: number;
  avg_expected_return_30d: number;
  avg_risk: number;
  momentum_20d: number;
  representative_assets: Array<{ ticker: string; name: string; type: string }>;
  rationale: string[];
  news_source: string;
  sample_headlines: string[];
};

export type RecommendationResponse = {
  market: Market;
  language: Language;
  trade_date: string;
  last_updated_at: string;
  selected_strategy: Strategy;
  market_state: MarketState;
  top_industries: IndustryRecommendation[];
  top_assets: AssetRecommendation[];
  mode: string;
};

export type PipelineStatus = {
  market: Market;
  language: Language;
  trade_date: string;
  last_updated_at: string;
  stages: PipelineStage[];
  model_checkpoint: string;
  mode: string;
};

export type AssetDetail = {
  ticker: string;
  trade_date: string;
  history_close: Array<{ date: string; close: number }>;
  rollout_path: Array<{ horizon: number; predicted_return: number; predicted_close?: number }>;
  features: {
    expected_return_30d: number;
    predicted_volatility: number;
    predicted_downside: number;
    bull_prob: number;
    sideway_prob: number;
    bear_prob: number;
  };
  explanation: string[];
  score: number;
  rank: number;
  sector: string;
  type: string;
};
