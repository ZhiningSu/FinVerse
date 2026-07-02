import type { AssetDetail, Language } from "@/types";

type RolloutChartProps = {
  asset?: AssetDetail | null;
  language: Language;
};

const COPY = {
  en: {
    empty: "Select an asset to inspect its rollout path.",
    eyebrow: "Rollout Viewer",
    titleSuffix: "imagined close-return path",
    score: "Score",
    predRet: "Pred Ret@30",
    predRisk: "Pred Risk",
    downside: "Downside",
  },
  zh: {
    empty: "请选择一只资产查看 rollout path。",
    eyebrow: "Rollout 查看器",
    titleSuffix: "close-return 想象路径",
    score: "评分",
    predRet: "预测30日收益",
    predRisk: "预测风险",
    downside: "下行风险",
  },
};

type Domain = { min: number; max: number };
type ChartPoint = { x: number; value: number };

function scaledY(value: number, height: number, domain: Domain) {
  const { min, max } = domain;
  const span = Math.max(max - min, 1e-6);
  return height - ((value - min) / span) * height;
}

function points(samples: ChartPoint[], height: number, domain: Domain) {
  return samples
    .map((sample) => `${sample.x.toFixed(2)},${scaledY(sample.value, height, domain).toFixed(2)}`)
    .join(" ");
}

function chartDomain(values: number[]): Domain {
  const finite = values.filter((value) => Number.isFinite(value));
  if (!finite.length) return { min: 0, max: 1 };
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const pad = Math.max((max - min) * 0.12, Math.abs(max) * 0.01, 1e-6);
  return { min: min - pad, max: max + pad };
}

export function RolloutChart({ asset, language }: RolloutChartProps) {
  const copy = COPY[language];
  if (!asset) {
    return (
      <section className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6 text-slate-400">
        {copy.empty}
      </section>
    );
  }

  const history = asset.history_close.slice(-30).map((row) => row.close);
  const historyLast = history[history.length - 1] ?? asset.rollout_path[0]?.predicted_close ?? 0;
  const predictedCloses = asset.rollout_path.map((row) => (
    row.predicted_close ?? historyLast * (1 + row.predicted_return)
  ));
  const rollout = [historyLast, ...predictedCloses];
  const domain = chartDomain([...history, ...rollout]);
  const width = 680;
  const height = 220;
  const splitX = width * 0.5;
  const historyPoints = history.map((value, index) => ({
    x: (index / Math.max(history.length - 1, 1)) * splitX,
    value,
  }));
  const rolloutPoints = rollout.map((value, index) => ({
    x: splitX + (index / Math.max(rollout.length - 1, 1)) * (width - splitX),
    value,
  }));
  const anchorY = scaledY(historyLast, height, domain);

  return (
    <section className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-slate-500">{copy.eyebrow}</p>
          <h2 className="mt-2 text-2xl font-semibold text-white">{asset.ticker} {copy.titleSuffix}</h2>
        </div>
        <div className="rounded-2xl bg-slate-950/70 px-4 py-3 text-right">
          <p className="text-xs text-slate-500">{copy.score}</p>
          <p className="text-xl font-semibold text-cyan-200">{asset.score?.toFixed(3)}</p>
        </div>
      </div>
      <svg className="mt-6 h-64 w-full overflow-visible" viewBox={`0 0 ${width} ${height}`}>
        <defs>
          <linearGradient id="rolloutGlow" x1="0" x2="1">
            <stop offset="0%" stopColor="#2DE2C5" />
            <stop offset="100%" stopColor="#7DD3FC" />
          </linearGradient>
        </defs>
        {[0, 1, 2, 3].map((line) => (
          <line
            key={line}
            x1="0"
            x2={width}
            y1={(height / 4) * line}
            y2={(height / 4) * line}
            stroke="rgba(148,163,184,0.16)"
          />
        ))}
        <line x1={splitX} x2={splitX} y1="0" y2={height} stroke="rgba(255,255,255,0.18)" strokeDasharray="4 6" />
        <polyline points={points(historyPoints, height, domain)} fill="none" stroke="#94A3B8" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        <polyline points={points(rolloutPoints, height, domain)} fill="none" stroke="url(#rolloutGlow)" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx={splitX} cy={anchorY} r="4.5" fill="#2DE2C5" stroke="#08111F" strokeWidth="2" />
      </svg>
      <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
        <div className="rounded-2xl bg-slate-950/60 p-3 text-slate-300">{copy.predRet}: {(asset.features.expected_return_30d * 100).toFixed(2)}%</div>
        <div className="rounded-2xl bg-slate-950/60 p-3 text-slate-300">{copy.predRisk}: {(asset.features.predicted_volatility * 100).toFixed(2)}%</div>
        <div className="rounded-2xl bg-slate-950/60 p-3 text-slate-300">{copy.downside}: {(asset.features.predicted_downside * 100).toFixed(2)}%</div>
      </div>
    </section>
  );
}
