import type { IndustryRecommendation, Language } from "@/types";

type IndustryPanelProps = {
  industries: IndustryRecommendation[];
  language: Language;
};

const COPY = {
  en: {
    eyebrow: "News + Macro Focus",
    title: "Top 5 industries",
    subtitle: "Ranked with sector momentum, risk, news sentiment, and macro-regime fit.",
    score: "Score",
    news: "News",
    macro: "Macro",
    reps: "Key assets",
    noNews: "neutral news signal",
    empty: "Industry recommendations will appear after the daily pipeline runs.",
  },
  zh: {
    eyebrow: "新闻 + 宏观焦点",
    title: "Top 5 推荐行业",
    subtitle: "综合行业动量、风险、新闻情绪和宏观市场阶段进行排序。",
    score: "评分",
    news: "新闻",
    macro: "宏观",
    reps: "代表资产",
    noNews: "新闻信号中性",
    empty: "每日 pipeline 运行后会显示行业推荐。",
  },
};

const RATIONALE_ZH: Record<string, string> = {
  "news sentiment is supportive": "新闻情绪较正面",
  "macro regime favors this industry": "宏观市场阶段支持该行业",
  "recent sector momentum is strong": "近期行业动量较强",
  "risk profile is relatively controlled": "风险特征相对可控",
  "balanced news, macro, return, and risk signals": "新闻、宏观、收益和风险信号较均衡",
};

function fmtPct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function fmtNews(value: number) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}`;
}

function reasonLabel(reason: string, language: Language) {
  return language === "zh" ? RATIONALE_ZH[reason] ?? reason : reason;
}

export function IndustryPanel({ industries, language }: IndustryPanelProps) {
  const copy = COPY[language];
  return (
    <section className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-5 shadow-2xl shadow-black/25">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-slate-500">{copy.eyebrow}</p>
          <h2 className="mt-2 text-2xl font-semibold text-white">{copy.title}</h2>
          <p className="mt-2 text-sm text-slate-400">{copy.subtitle}</p>
        </div>
      </div>
      {industries.length === 0 ? (
        <p className="rounded-2xl bg-white/[0.04] p-4 text-sm text-slate-400">{copy.empty}</p>
      ) : (
        <div className="grid gap-3 lg:grid-cols-5">
          {industries.map((industry) => (
            <article key={industry.sector} className="rounded-3xl border border-white/10 bg-white/[0.04] p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs text-slate-500">#{industry.rank}</p>
                  <h3 className="mt-1 text-lg font-semibold text-white">{industry.sector}</h3>
                </div>
                <span className="rounded-full bg-cyan-300/10 px-2.5 py-1 text-xs text-cyan-100">
                  {copy.score} {industry.score.toFixed(3)}
                </span>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-2xl bg-slate-950/60 p-2 text-slate-300">
                  {copy.news}: {industry.news_count > 0 ? fmtNews(industry.news_score) : copy.noNews}
                </div>
                <div className="rounded-2xl bg-slate-950/60 p-2 text-slate-300">
                  {copy.macro}: {fmtPct(industry.macro_score)}
                </div>
              </div>
              <div className="mt-3 space-y-2 text-xs text-slate-400">
                {industry.rationale.slice(0, 2).map((reason) => (
                  <p key={reason} className="rounded-2xl bg-slate-950/40 px-3 py-2">
                    {reasonLabel(reason, language)}
                  </p>
                ))}
              </div>
              <p className="mt-3 text-xs text-slate-500">{copy.reps}</p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {industry.representative_assets.slice(0, 3).map((asset) => (
                  <span key={asset.ticker} className="rounded-full bg-teal-300/10 px-2 py-1 text-xs text-teal-100">
                    {asset.ticker}
                  </span>
                ))}
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
