import { Clock3, RadioTower } from "lucide-react";
import type { Language, LiveQuotesResponse } from "@/types";

type LiveQuotePanelProps = {
  data?: LiveQuotesResponse | null;
  language: Language;
  loading?: boolean;
  error?: string | null;
  refreshIntervalMs?: number;
};

const COPY = {
  en: {
    eyebrow: "Live Market Data",
    title: "Real-time quote monitor",
    subtitle: "Read-only live quotes for the current Top 20 universe. Falls back to the latest daily snapshot when the live source is unavailable.",
    loading: "Loading live quotes...",
    empty: "Live quotes are not available yet.",
    realtime: "live source",
    snapshot: "daily snapshot fallback",
    price: "Price",
    change: "Change",
    asOf: "As of",
    source: "Source",
    autoRefresh: "Auto refresh",
  },
  zh: {
    eyebrow: "实盘行情",
    title: "实时行情监控",
    subtitle: "当前 Top 20 资产池的只读实时行情；实时源不可用时自动回退到最新日频快照。",
    loading: "正在加载实时行情...",
    empty: "实时行情暂不可用。",
    realtime: "实时行情源",
    snapshot: "日频快照回退",
    price: "价格",
    change: "涨跌幅",
    asOf: "截至",
    source: "来源",
    autoRefresh: "自动刷新",
  },
};

function fmtPrice(value: number | null, currency: string) {
  if (value === null || Number.isNaN(value)) return "--";
  const digits = currency === "CNY" ? 2 : 2;
  return `${value.toFixed(digits)} ${currency}`;
}

function fmtPct(value: number | null) {
  if (value === null || Number.isNaN(value)) return "--";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}%`;
}

function tone(value: number | null) {
  if (value === null) return "text-slate-400";
  if (value > 0) return "text-teal-200";
  if (value < 0) return "text-rose-200";
  return "text-slate-300";
}

export function LiveQuotePanel({ data, language, loading = false, error = null, refreshIntervalMs }: LiveQuotePanelProps) {
  const copy = COPY[language];
  const quotes = data?.quotes ?? [];
  const statusLabel = data?.is_realtime ? copy.realtime : copy.snapshot;
  const refreshSeconds = refreshIntervalMs ? Math.round(refreshIntervalMs / 1000) : null;

  return (
    <section className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-5 shadow-2xl shadow-black/25">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="flex items-center gap-2 text-xs uppercase tracking-[0.25em] text-cyan-200">
            <RadioTower className="h-4 w-4" />
            {copy.eyebrow}
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-white">{copy.title}</h2>
          <p className="mt-2 max-w-4xl text-sm text-slate-400">{copy.subtitle}</p>
        </div>
        <div className="rounded-full border border-cyan-300/20 bg-cyan-300/10 px-4 py-2 text-xs text-cyan-100">
          {statusLabel}{refreshSeconds ? ` · ${copy.autoRefresh} ${refreshSeconds}s` : ""}
        </div>
      </div>

      {loading ? (
        <p className="rounded-2xl bg-white/[0.04] p-4 text-sm text-slate-400">{copy.loading}</p>
      ) : error ? (
        <p className="rounded-2xl border border-amber-300/20 bg-amber-300/10 p-4 text-sm text-amber-100">{error}</p>
      ) : quotes.length === 0 ? (
        <p className="rounded-2xl bg-white/[0.04] p-4 text-sm text-slate-400">{copy.empty}</p>
      ) : (
        <>
          <div className="grid max-h-[360px] gap-3 overflow-auto pr-1 md:grid-cols-2 xl:grid-cols-4">
            {quotes.map((quote) => (
              <article key={quote.ticker} className="rounded-3xl border border-white/10 bg-white/[0.04] p-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-semibold text-cyan-100">{quote.ticker}</p>
                    <p className="mt-1 line-clamp-1 text-xs text-slate-500">{quote.name}</p>
                  </div>
                  <span className={`text-sm font-semibold ${tone(quote.change_percent)}`}>
                    {fmtPct(quote.change_percent)}
                  </span>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                  <div className="rounded-2xl bg-slate-950/60 p-2">
                    <p className="text-slate-500">{copy.price}</p>
                    <p className="mt-1 font-semibold text-white">{fmtPrice(quote.price, quote.currency)}</p>
                  </div>
                  <div className="rounded-2xl bg-slate-950/60 p-2">
                    <p className="text-slate-500">{copy.change}</p>
                    <p className={`mt-1 font-semibold ${tone(quote.change)}`}>
                      {quote.change === null ? "--" : `${quote.change > 0 ? "+" : ""}${quote.change.toFixed(2)}`}
                    </p>
                  </div>
                </div>
              </article>
            ))}
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-slate-500">
            <span className="inline-flex items-center gap-1">
              <Clock3 className="h-3.5 w-3.5" />
              {copy.asOf}: {data?.as_of ?? "--"}
            </span>
            <span>{copy.source}: {data?.source ?? "--"}</span>
          </div>
        </>
      )}
    </section>
  );
}
