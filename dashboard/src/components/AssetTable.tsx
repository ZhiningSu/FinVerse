import type { AssetRecommendation, Language } from "@/types";

type AssetTableProps = {
  assets: AssetRecommendation[];
  onSelect: (ticker: string) => void;
  language: Language;
};

const fmtPct = (value: number) => `${(value * 100).toFixed(2)}%`;

const COPY = {
  en: {
    eyebrow: "Top-K Universe",
    title: "Today's stock / fund picks",
    helper: "Click ticker to inspect rollout details",
    rank: "Rank",
    ticker: "Ticker",
    name: "Name",
    sector: "Sector",
    score: "Score",
    ret30: "Ret@30",
    risk: "Risk",
    reason: "Reason",
    stock: "STOCK",
    etf: "ETF",
    fallback: "balanced score",
  },
  zh: {
    eyebrow: "Top-K 资产池",
    title: "今日中国股票 / 基金推荐",
    helper: "点击代码查看 rollout 详情",
    rank: "排名",
    ticker: "代码",
    name: "名称",
    sector: "行业",
    score: "评分",
    ret30: "30日收益",
    risk: "风险",
    reason: "推荐理由",
    stock: "股票",
    etf: "ETF",
    fallback: "收益与风险综合评分均衡",
  },
};

const REASON_ZH: Record<string, string> = {
  "predicted upside ranks high": "预测上行空间排名较高",
  "risk estimate is relatively low": "风险估计相对较低",
  "downside estimate is controlled": "下行风险较可控",
  "sector matches the selected strategy": "行业与当前策略匹配",
  "ETF exposure improves resilience": "ETF 配置增强组合韧性",
  "balanced score across return and risk features": "收益与风险特征综合评分均衡",
  "balanced score": "收益与风险综合评分均衡",
};

function reasonLabel(reason: string | undefined, language: Language, fallback: string) {
  if (!reason) return fallback;
  return language === "zh" ? REASON_ZH[reason] ?? reason : reason;
}

function typeLabel(type: AssetRecommendation["type"], language: Language) {
  if (language === "zh") return type === "etf" ? COPY.zh.etf : COPY.zh.stock;
  return type === "etf" ? COPY.en.etf : COPY.en.stock;
}

export function AssetTable({ assets, onSelect, language }: AssetTableProps) {
  const copy = COPY[language];
  return (
    <section className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-5 shadow-2xl shadow-black/25">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-slate-500">{copy.eyebrow}</p>
          <h2 className="mt-2 text-2xl font-semibold text-white">{copy.title}</h2>
        </div>
        <p className="text-sm text-slate-400">{copy.helper}</p>
      </div>
      <div className="overflow-hidden rounded-2xl border border-white/10">
        <table className="w-full border-collapse text-left text-sm">
          <thead className="bg-white/[0.06] text-xs uppercase tracking-[0.18em] text-slate-400">
            <tr>
              <th className="px-4 py-3">{copy.rank}</th>
              <th className="px-4 py-3">{copy.ticker}</th>
              <th className="px-4 py-3">{copy.name}</th>
              <th className="px-4 py-3">{copy.sector}</th>
              <th className="px-4 py-3">{copy.score}</th>
              <th className="px-4 py-3">{copy.ret30}</th>
              <th className="px-4 py-3">{copy.risk}</th>
              <th className="px-4 py-3">{copy.reason}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/10">
            {assets.map((asset) => (
              <tr key={asset.ticker} className="bg-slate-950/20 transition hover:bg-cyan-300/10">
                <td className="px-4 py-4 text-slate-400">#{asset.rank}</td>
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
                <td className="px-4 py-4 text-slate-300">{asset.sector}</td>
                <td className="px-4 py-4 text-white">{asset.score.toFixed(3)}</td>
                <td className="px-4 py-4 text-teal-200">{fmtPct(asset.expected_return_30d)}</td>
                <td className="px-4 py-4 text-amber-100">{fmtPct(asset.predicted_volatility)}</td>
                <td className="px-4 py-4 text-slate-400">{reasonLabel(asset.reasons?.[0], language, copy.fallback)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
