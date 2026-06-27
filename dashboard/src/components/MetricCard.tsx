type MetricCardProps = {
  label: string;
  value: string;
  tone?: "cyan" | "amber" | "rose" | "slate";
  detail?: string;
};

const toneClass = {
  cyan: "border-cyan-300/25 bg-cyan-300/10 text-cyan-100",
  amber: "border-amber-300/25 bg-amber-300/10 text-amber-100",
  rose: "border-rose-300/25 bg-rose-300/10 text-rose-100",
  slate: "border-slate-300/15 bg-slate-400/10 text-slate-100",
};

export function MetricCard({ label, value, tone = "slate", detail }: MetricCardProps) {
  return (
    <article className={`rounded-3xl border p-5 shadow-2xl shadow-black/20 ${toneClass[tone]}`}>
      <p className="text-xs uppercase tracking-[0.28em] text-slate-400">{label}</p>
      <p className="mt-3 text-3xl font-semibold tracking-tight text-white">{value}</p>
      {detail ? <p className="mt-2 text-sm text-slate-400">{detail}</p> : null}
    </article>
  );
}
