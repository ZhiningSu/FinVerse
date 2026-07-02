import type { AssetRecommendation, AssetSortMode, Language } from "@/types";

type AssetTableProps = {
  assets: AssetRecommendation[];
  onSelect: (ticker: string) => void;
  language: Language;
  selectedTicker?: string;
  sortMode: AssetSortMode;
  onSortModeChange: (mode: AssetSortMode) => void;
};

const fmtPct = (value: number) => `${(value * 100).toFixed(2)}%`;

const COPY = {
  en: {
    eyebrow: "Top-20 Universe",
    title: "Live-adjusted Top 20 stock / fund picks",
    helper: "Click ticker to inspect rollout details",
    rank: "Rank",
    modelRank: "Model",
    strategyRank: "Strategy",
    modelSort: "Model ranking",
    strategySort: "Strategy ranking",
    modelScore: "Model Ret",
    strategyScore: "Strategy Score",
    ticker: "Ticker",
    name: "Name",
    sector: "Sector",
    score: "Live Score",
    live: "Live",
    ret30: "Ret@30",
    risk: "Risk",
    reason: "Reason",
    stock: "STOCK",
    etf: "ETF",
    fallback: "balanced score",
    hotTheme: "hot tech theme",
    positiveMomentum: "live momentum is positive",
  },
  zh: {
    eyebrow: "Top-20 资产池",
    title: "实时调整 Top 20 股票 / 基金推荐",
    helper: "点击代码查看 rollout 详情",
    rank: "排名",
    modelRank: "模型排名",
    strategyRank: "策略排名",
    modelSort: "模型排序",
    strategySort: "策略排序",
    modelScore: "模型收益",
    strategyScore: "策略评分",
    ticker: "代码",
    name: "名称",
    sector: "行业",
    score: "实时评分",
    live: "实时涨跌",
    ret30: "30日收益",
    risk: "风险",
    reason: "推荐理由",
    stock: "股票",
    etf: "ETF",
    fallback: "收益与风险综合评分均衡",
    hotTheme: "科技热点主题",
    positiveMomentum: "实时动量为正",
  },
};

const REASON_ZH: Record<string, string> = {
  "live momentum is positive": "实时动量为正",
  "hot technology theme exposure": "科技/芯片/存储等热点主题暴露",
  "predicted upside ranks high": "预测上行空间排名较高",
  "risk estimate is relatively low": "风险估计相对较低",
  "downside estimate is controlled": "下行风险较可控",
  "sector matches the selected strategy": "行业与当前策略匹配",
  "ETF exposure improves resilience": "ETF 配置增强组合韧性",
  "news/theme heat is elevated": "新闻/主题热度较高",
  "balanced score across return and risk features": "收益与风险特征综合评分均衡",
  "balanced score": "收益与风险综合评分均衡",
};

const SECTOR_LABELS: Record<string, { en: string; zh: string }> = {
  Technology: { en: "Technology", zh: "科技" },
  "Communication Services": { en: "Communication Services", zh: "通信服务" },
  "Consumer Discretionary": { en: "Consumer Discretionary", zh: "可选消费" },
  "Consumer Staples": { en: "Consumer Staples", zh: "必选消费" },
  Financials: { en: "Financials", zh: "金融" },
  Healthcare: { en: "Healthcare", zh: "医药" },
  Energy: { en: "Energy", zh: "能源" },
  Industrials: { en: "Industrials", zh: "工业" },
  Utilities: { en: "Utilities", zh: "公用事业" },
  Materials: { en: "Materials", zh: "材料" },
  "Real Estate": { en: "Real Estate", zh: "地产" },
  "Market ETF": { en: "Market ETF", zh: "宽基ETF" },
  "Sector ETF": { en: "Sector ETF", zh: "行业ETF" },
  科技: { en: "Technology", zh: "科技" },
  通信: { en: "Communication Services", zh: "通信" },
  消费: { en: "Consumer", zh: "消费" },
  金融: { en: "Financials", zh: "金融" },
  医药: { en: "Healthcare", zh: "医药" },
  能源: { en: "Energy", zh: "能源" },
  新能源: { en: "New Energy", zh: "新能源" },
  公用事业: { en: "Utilities", zh: "公用事业" },
  材料: { en: "Materials", zh: "材料" },
  地产: { en: "Real Estate", zh: "地产" },
  工业: { en: "Industrials", zh: "工业" },
  宽基ETF: { en: "Market ETF", zh: "宽基ETF" },
  行业ETF: { en: "Sector ETF", zh: "行业ETF" },
};

function reasonLabel(reason: string | undefined, language: Language, fallback: string) {
  if (!reason) return fallback;
  return language === "zh" ? REASON_ZH[reason] ?? reason : reason;
}

function sectorLabel(sector: string, language: Language) {
  return SECTOR_LABELS[sector]?.[language] ?? sector;
}

function typeLabel(type: AssetRecommendation["type"], language: Language) {
  if (language === "zh") return type === "etf" ? COPY.zh.etf : COPY.zh.stock;
  return type === "etf" ? COPY.en.etf : COPY.en.stock;
}

function fmtMaybePct(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}%`;
}

function liveTone(value: number | null | undefined) {
  if (value === null || value === undefined) return "text-slate-400";
  if (value > 0) return "text-teal-200";
  if (value < 0) return "text-rose-200";
  return "text-slate-300";
}

export function AssetTable({
  assets,
  onSelect,
  language,
  selectedTicker,
  sortMode,
  onSortModeChange,
}: AssetTableProps) {
  const copy = COPY[language];
  const sortModes: Array<{ mode: AssetSortMode; label: string }> = [
    { mode: "model", label: copy.modelSort },
    { mode: "strategy", label: copy.strategySort },
  ];
  const peerRankLabel = sortMode === "model" ? copy.strategyRank : copy.modelRank;
  const scoreLabel = sortMode === "model" ? copy.modelScore : copy.strategyScore;
  return (
    <section className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-5 shadow-2xl shadow-black/25">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-slate-500">{copy.eyebrow}</p>
          <h2 className="mt-2 text-2xl font-semibold text-white">{copy.title}</h2>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex rounded-full border border-white/10 bg-slate-900/80 p-1 text-xs">
            {sortModes.map((item) => (
              <button
                key={item.mode}
                onClick={() => onSortModeChange(item.mode)}
                className={`rounded-full px-4 py-2 transition ${sortMode === item.mode ? "bg-cyan-300/20 text-cyan-100" : "text-slate-400 hover:text-slate-100"}`}
              >
                {item.label}
              </button>
            ))}
          </div>
          <p className="text-sm text-slate-400">{copy.helper}</p>
        </div>
      </div>
      <div className="max-h-[820px] overflow-auto rounded-2xl border border-white/10">
        <table className="min-w-[1120px] w-full border-collapse text-left text-sm">
          <thead className="bg-white/[0.06] text-xs uppercase tracking-[0.18em] text-slate-400">
            <tr>
              <th className="px-4 py-3">{copy.rank}</th>
              <th className="px-4 py-3">{peerRankLabel}</th>
              <th className="px-4 py-3">{copy.ticker}</th>
              <th className="px-4 py-3">{copy.name}</th>
              <th className="px-4 py-3">{copy.sector}</th>
              <th className="px-4 py-3">{scoreLabel}</th>
              <th className="px-4 py-3">{copy.live}</th>
              <th className="px-4 py-3">{copy.ret30}</th>
              <th className="px-4 py-3">{copy.risk}</th>
              <th className="px-4 py-3">{copy.reason}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/10">
            {assets.map((asset) => {
              const selected = selectedTicker?.toUpperCase() === asset.ticker.toUpperCase();
              const peerRank = sortMode === "model" ? asset.strategy_rank : asset.model_rank;
              const scoreValue = sortMode === "model"
                ? fmtPct(asset.model_sort_score ?? asset.expected_return_30d)
                : (asset.strategy_sort_score ?? asset.score).toFixed(3);
              return (
                <tr
                  key={asset.ticker}
                  className={`${selected ? "bg-cyan-300/15 ring-1 ring-inset ring-cyan-300/30" : "bg-slate-950/20"} transition hover:bg-cyan-300/10`}
                >
                  <td className="px-4 py-4 text-slate-400">#{asset.rank}</td>
                  <td className="px-4 py-4 text-slate-500">#{peerRank ?? asset.rank}</td>
                  <td className="px-4 py-4">
                    <button
                      className="font-semibold text-cyan-200 underline-offset-4 hover:text-cyan-100 hover:underline"
                      onClick={() => onSelect(asset.ticker)}
                    >
                      {asset.ticker}
                    </button>
                    <p className="text-xs text-slate-500">{typeLabel(asset.type, language)}</p>
                  </td>
                  <td className="max-w-[9rem] px-4 py-4 text-slate-200">
                    <span className="line-clamp-2">{asset.name}</span>
                  </td>
                  <td className="px-4 py-4 text-slate-300">{sectorLabel(asset.sector, language)}</td>
                  <td className="px-4 py-4 text-white">
                    {scoreValue}
                    {asset.hot_theme ? (
                      <p className="mt-1 text-xs font-normal text-amber-200">{copy.hotTheme}</p>
                    ) : null}
                  </td>
                  <td className={`px-4 py-4 font-semibold ${liveTone(asset.live_change_percent)}`}>
                    {fmtMaybePct(asset.live_change_percent)}
                  </td>
                  <td className="px-4 py-4 text-teal-200">{fmtPct(asset.expected_return_30d)}</td>
                  <td className="px-4 py-4 text-amber-100">{fmtPct(asset.predicted_volatility)}</td>
                  <td className="px-4 py-4 text-slate-400">{reasonLabel(asset.reasons?.[0], language, copy.fallback)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
