import type { Language, RegimeProbs } from "@/types";

type RegimeGaugeProps = {
  probs: RegimeProbs;
  language: Language;
};

const COPY = {
  en: {
    eyebrow: "Market Regime",
    title: "Probability distribution",
    adapter: "live adapter",
    bear: "Bear",
    sideway: "Sideway",
    bull: "Bull",
  },
  zh: {
    eyebrow: "市场阶段",
    title: "概率分布",
    adapter: "实时 adapter",
    bear: "熊市",
    sideway: "震荡",
    bull: "牛市",
  },
};

export function RegimeGauge({ probs, language }: RegimeGaugeProps) {
  const copy = COPY[language];
  const segments = [
    { name: copy.bear, value: probs.bear, color: "bg-rose-400" },
    { name: copy.sideway, value: probs.sideway, color: "bg-slate-300" },
    { name: copy.bull, value: probs.bull, color: "bg-teal-300" },
  ];

  return (
    <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-slate-400">{copy.eyebrow}</p>
          <h3 className="mt-2 text-xl font-semibold text-white">{copy.title}</h3>
        </div>
        <div className="rounded-full border border-teal-300/30 px-3 py-1 text-xs text-teal-200">{copy.adapter}</div>
      </div>
      <div className="mt-5 h-3 overflow-hidden rounded-full bg-slate-800">
        <div className="flex h-full">
          {segments.map((segment) => (
            <span
              key={segment.name}
              className={segment.color}
              style={{ width: `${Math.max(segment.value * 100, 2)}%` }}
            />
          ))}
        </div>
      </div>
      <div className="mt-5 grid grid-cols-3 gap-3">
        {segments.map((segment) => (
          <div key={segment.name} className="rounded-2xl bg-slate-950/60 p-3">
            <p className="text-xs text-slate-500">{segment.name}</p>
            <p className="mt-1 text-lg font-semibold text-white">{(segment.value * 100).toFixed(1)}%</p>
          </div>
        ))}
      </div>
    </div>
  );
}
