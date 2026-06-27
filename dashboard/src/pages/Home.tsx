import { useCallback, useEffect, useState } from "react";
import { Activity, BrainCircuit, RefreshCw, ShieldCheck, Sparkles } from "lucide-react";
import { AssetTable } from "@/components/AssetTable";
import { IndustryPanel } from "@/components/IndustryPanel";
import { MetricCard } from "@/components/MetricCard";
import { PipelinePanel } from "@/components/PipelinePanel";
import { RegimeGauge } from "@/components/RegimeGauge";
import { RolloutChart } from "@/components/RolloutChart";
import { getAssetDetail, getLatestRecommendation, getPipelineStatus, runPipeline } from "@/lib/api";
import type { AssetDetail, Language, Market, PipelineStatus, RecommendationResponse, Strategy } from "@/types";

type HomeProps = {
  initialMarket?: Market;
  initialLanguage?: Language;
};

const COPY = {
  en: {
    loading: "Loading the FinVerse dynamic pipeline...",
    loadError: "Unable to load the latest dashboard data. Please run the pipeline again.",
    eyebrow: "FinVerse live agent tool",
    title: "FinVerse Market Intelligence",
    description: "Daily market state, strategy style, and stock / ETF recommendations across U.S. and China markets. This is a decision-support dashboard, not an automated trading system.",
    github: "GitHub Repository",
    language: "Language",
    market: "Market",
    english: "English",
    chinese: "中文",
    usMarket: "U.S. Market",
    cnMarket: "China Market",
    refresh: "Run today's pipeline",
    tradeDate: "Trade Date",
    strategy: "Strategy",
    marketRegime: "Market Regime",
    topAssets: "Top Assets",
    return20d: "20d return",
    rankedDetail: "stocks / ETFs ranked",
    selectedStrategy: "Selected strategy",
    confidence: "confidence",
    diagnosticOnly: "diagnostic only",
    noExecution: "no trade execution",
    riskTitle: "Explanation & Risk Notice",
    selectAsset: "Select an asset to show explanations.",
    disclaimer: "This tool provides world-model-guided decision support and is not investment advice.",
    lastUpdated: "Last updated",
  },
  zh: {
    loading: "正在加载 FinVerse 动态 pipeline...",
    loadError: "无法加载最新 dashboard 数据，请重新运行 pipeline。",
    eyebrow: "FinVerse 动态金融 Agent",
    title: "FinVerse Market Intelligence",
    description: "每日更新中美市场状态、策略风格和股票 / 基金推荐。当前阶段是 decision-support dashboard，不是自动交易系统。",
    github: "GitHub 项目链接",
    language: "语言",
    market: "市场",
    english: "English",
    chinese: "中文",
    usMarket: "美国市场",
    cnMarket: "中国市场",
    refresh: "手动刷新 pipeline",
    tradeDate: "交易日期",
    strategy: "策略",
    marketRegime: "市场阶段",
    topAssets: "推荐资产",
    return20d: "20日收益",
    rankedDetail: "股票 / ETF 已排序",
    selectedStrategy: "选中的策略",
    confidence: "置信度",
    diagnosticOnly: "仅作诊断",
    noExecution: "不执行交易",
    riskTitle: "解释与风险提示",
    selectAsset: "选择资产后显示解释。",
    disclaimer: "该工具展示 world-model-guided decision support，不构成投资建议。",
    lastUpdated: "最后更新",
  },
};

const STRATEGY_COPY: Record<string, Record<Language, { name: string; description: string }>> = {
  "Aggressive Growth": {
    en: { name: "Aggressive Growth", description: "Prioritizes assets with stronger predicted upside and higher bull-regime probability." },
    zh: { name: "进攻成长", description: "偏向高预测收益和高牛市概率的资产。" },
  },
  "Balanced Growth": {
    en: { name: "Balanced Growth", description: "Balances predicted return, risk control, and recent momentum." },
    zh: { name: "均衡成长", description: "在预测收益、风险控制和近期动量之间折中。" },
  },
  "Defensive Quality": {
    en: { name: "Defensive Quality", description: "Favors lower-risk assets with controlled downside and defensive sector exposure." },
    zh: { name: "防御质量", description: "偏向较低风险、较低 downside 和防御型行业。" },
  },
  "Crisis Resilience": {
    en: { name: "Crisis Resilience", description: "Emphasizes ETF and defensive exposure when market stress is elevated." },
    zh: { name: "危机韧性", description: "在市场压力较高时偏向 ETF 和防御型资产，强调 downside 控制。" },
  },
};

const REGIME_COPY: Record<string, Record<Language, string>> = {
  Bull: { en: "Bull", zh: "牛市" },
  Bear: { en: "Bear", zh: "熊市" },
  Sideway: { en: "Sideway", zh: "震荡" },
};

const REASON_ZH: Record<string, string> = {
  "predicted upside ranks high": "预测上行空间排名较高",
  "risk estimate is relatively low": "风险估计相对较低",
  "downside estimate is controlled": "下行风险较可控",
  "sector matches the selected strategy": "行业与当前策略匹配",
  "ETF exposure improves resilience": "ETF 配置增强组合韧性",
  "balanced score across return and risk features": "收益与风险特征综合评分均衡",
};

function displayStrategy(strategy: Strategy, language: Language) {
  return STRATEGY_COPY[strategy.name]?.[language] ?? strategy;
}

function displayRegime(regime: string, language: Language) {
  return REGIME_COPY[regime]?.[language] ?? regime;
}

function displayReason(reason: string, language: Language) {
  return language === "zh" ? REASON_ZH[reason] ?? reason : reason;
}

export default function Home({ initialMarket = "us", initialLanguage = "en" }: HomeProps) {
  const [language, setLanguage] = useState<Language>(initialLanguage);
  const [marketId, setMarketId] = useState<Market>(initialMarket);
  const [recommendation, setRecommendation] = useState<RecommendationResponse | null>(null);
  const [pipeline, setPipeline] = useState<PipelineStatus | null>(null);
  const [asset, setAsset] = useState<AssetDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const copy = COPY[language];
  const activeMarketLabel = marketId === "us" ? copy.usMarket : copy.cnMarket;

  const refresh = useCallback(async (preferredTicker?: string) => {
    setLoading(true);
    setError(null);
    const [rec, status] = await Promise.all([getLatestRecommendation(marketId), getPipelineStatus(marketId)]);
    setRecommendation(rec);
    setPipeline(status);
    const selectedTicker = preferredTicker ?? rec.top_assets[0]?.ticker;
    if (selectedTicker) {
      setAsset(await getAssetDetail(selectedTicker, marketId));
    } else {
      setAsset(null);
    }
    setLoading(false);
  }, [marketId]);

  async function selectAsset(ticker: string) {
    setAsset(await getAssetDetail(ticker, marketId));
  }

  async function triggerRun() {
    await runPipeline(marketId);
    window.setTimeout(() => {
      refresh(asset?.ticker).catch((err) => {
        setError(err instanceof Error ? err.message : copy.loadError);
        setLoading(false);
      });
    }, 800);
  }

  useEffect(() => {
    setRecommendation(null);
    setPipeline(null);
    setAsset(null);
    refresh().catch((err) => {
      setError(err instanceof Error ? err.message : COPY.en.loadError);
      setLoading(false);
    });
  }, [refresh]);

  if (loading || !recommendation || !pipeline) {
    return <div className="min-h-screen bg-[#08111F] p-10 text-slate-300">{error ?? copy.loading}</div>;
  }

  const market = recommendation.market_state;
  const strategy = recommendation.selected_strategy;
  const localizedStrategy = displayStrategy(strategy, language);
  const localizedRegime = displayRegime(market.regime, language);
  const explanations = asset?.explanation?.length ? asset.explanation.map((item) => displayReason(item, language)) : [copy.selectAsset];

  return (
    <main className="min-h-screen overflow-hidden bg-[#08111F] text-slate-100">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_20%_10%,rgba(45,226,197,0.18),transparent_28%),radial-gradient(circle_at_85%_15%,rgba(255,184,77,0.12),transparent_26%),linear-gradient(135deg,rgba(255,255,255,0.05),transparent_35%)]" />
      <div className="relative mx-auto max-w-7xl px-6 py-8">
        <header className="border-b border-white/10 pb-8 text-center">
          <div className="mx-auto flex max-w-4xl flex-col items-center">
            <p className="flex items-center justify-center gap-2 text-xs uppercase tracking-[0.35em] text-cyan-200">
              <BrainCircuit className="h-4 w-4" />
              {copy.eyebrow}
            </p>
            <h1 className="mt-4 text-4xl font-semibold tracking-tight text-white md:text-6xl">
              {copy.title}
            </h1>
            <a
              href="https://github.com/ZhiningSu/FinVerse"
              target="_blank"
              rel="noreferrer"
              className="mt-3 text-sm font-medium text-cyan-200 underline-offset-4 transition hover:text-cyan-100 hover:underline"
            >
              {copy.github}: github.com/ZhiningSu/FinVerse
            </a>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
              {copy.description}
            </p>
            <div className="mt-4 inline-flex rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-xs text-cyan-100">
              {activeMarketLabel} · {copy[language === "en" ? "english" : "chinese"]}
            </div>
          </div>
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <div className="flex items-center gap-2 rounded-full border border-white/10 bg-slate-950/70 p-1 text-sm">
              <span className="pl-3 text-xs text-slate-500">{copy.language}</span>
              <button
                onClick={() => setLanguage("en")}
                className={`rounded-full px-4 py-2 transition ${language === "en" ? "bg-cyan-300/20 text-cyan-100" : "text-slate-400 hover:text-slate-100"}`}
              >
                {copy.english}
              </button>
              <button
                onClick={() => setLanguage("zh")}
                className={`rounded-full px-4 py-2 transition ${language === "zh" ? "bg-cyan-300/20 text-cyan-100" : "text-slate-400 hover:text-slate-100"}`}
              >
                {copy.chinese}
              </button>
            </div>
            <div className="flex items-center gap-2 rounded-full border border-white/10 bg-slate-950/70 p-1 text-sm">
              <span className="pl-3 text-xs text-slate-500">{copy.market}</span>
              <button
                onClick={() => setMarketId("us")}
                className={`rounded-full px-4 py-2 transition ${marketId === "us" ? "bg-cyan-300/20 text-cyan-100" : "text-slate-400 hover:text-slate-100"}`}
              >
                {copy.usMarket}
              </button>
              <button
                onClick={() => setMarketId("cn")}
                className={`rounded-full px-4 py-2 transition ${marketId === "cn" ? "bg-cyan-300/20 text-cyan-100" : "text-slate-400 hover:text-slate-100"}`}
              >
                {copy.cnMarket}
              </button>
            </div>
            <button
              onClick={triggerRun}
              className="inline-flex items-center gap-2 rounded-full border border-cyan-300/30 bg-cyan-300/10 px-5 py-3 text-sm font-medium text-cyan-100 transition hover:bg-cyan-300/20"
            >
              <RefreshCw className="h-4 w-4" />
              {copy.refresh}
            </button>
          </div>
        </header>

        <section className="mt-8 grid gap-4 md:grid-cols-4">
          <MetricCard label={copy.tradeDate} value={recommendation.trade_date} detail={recommendation.mode} tone="cyan" />
          <MetricCard label={copy.strategy} value={localizedStrategy.name} detail={localizedStrategy.description} tone="amber" />
          <MetricCard label={copy.marketRegime} value={localizedRegime} detail={`${copy.return20d} ${(market.market_return_20d * 100).toFixed(2)}%`} tone="slate" />
          <MetricCard label={copy.topAssets} value={String(recommendation.top_assets.length)} detail={copy.rankedDetail} tone="cyan" />
        </section>

        <section className="mt-6 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-6">
            <div className="flex items-center gap-3">
              <Sparkles className="h-5 w-5 text-amber-200" />
              <div>
                <p className="text-xs uppercase tracking-[0.25em] text-slate-500">{copy.selectedStrategy}</p>
                <h2 className="mt-1 text-3xl font-semibold text-white">{localizedStrategy.name}</h2>
              </div>
            </div>
            <p className="mt-4 leading-7 text-slate-300">{localizedStrategy.description}</p>
            <div className="mt-5 flex flex-wrap gap-3 text-sm">
              <span className="rounded-full bg-teal-300/10 px-3 py-2 text-teal-100">
                {copy.confidence} {(strategy.confidence * 100).toFixed(1)}%
              </span>
              <span className="rounded-full bg-amber-300/10 px-3 py-2 text-amber-100">{copy.diagnosticOnly}</span>
              <span className="rounded-full bg-slate-300/10 px-3 py-2 text-slate-200">{copy.noExecution}</span>
            </div>
          </div>
          <RegimeGauge probs={market.regime_probs} language={language} />
        </section>

        <section className="mt-6">
          <IndustryPanel industries={recommendation.top_industries ?? []} language={language} />
        </section>

        <section className="mt-6 grid items-start gap-6 xl:grid-cols-[1.35fr_0.95fr]">
          <AssetTable
            assets={recommendation.top_assets}
            onSelect={selectAsset}
            language={language}
            selectedTicker={asset?.ticker}
          />
          <aside className="space-y-6 xl:sticky xl:top-6">
            <RolloutChart asset={asset} language={language} />
            <div className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6">
              <div className="flex items-center gap-3">
                <ShieldCheck className="h-5 w-5 text-teal-200" />
                <h2 className="text-xl font-semibold text-white">{copy.riskTitle}</h2>
              </div>
              <div className="mt-5 space-y-3 text-sm text-slate-300">
                {explanations.map((item) => (
                  <p key={item} className="rounded-2xl bg-slate-950/60 p-3">{item}</p>
                ))}
                <p className="rounded-2xl border border-amber-300/20 bg-amber-300/10 p-3 text-amber-100">
                  {copy.disclaimer}
                </p>
              </div>
            </div>
          </aside>
        </section>

        <section className="mt-6">
          <PipelinePanel stages={pipeline.stages} language={language} />
        </section>

        <footer className="mt-10 flex items-center gap-2 border-t border-white/10 py-6 text-xs text-slate-500">
          <Activity className="h-4 w-4" />
          {copy.lastUpdated}: {recommendation.last_updated_at}
        </footer>
      </div>
    </main>
  );
}
