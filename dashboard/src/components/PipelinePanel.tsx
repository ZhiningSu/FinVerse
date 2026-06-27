import { CheckCircle2, CircleDashed, XCircle } from "lucide-react";
import type { Language, PipelineStage } from "@/types";

type PipelinePanelProps = {
  stages: PipelineStage[];
  language: Language;
};

function icon(status: string) {
  if (status === "success") return <CheckCircle2 className="h-4 w-4 text-teal-300" />;
  if (status === "failed") return <XCircle className="h-4 w-4 text-rose-300" />;
  return <CircleDashed className="h-4 w-4 text-amber-200" />;
}

const COPY = {
  en: {
    eyebrow: "Pipeline Monitor",
    title: "Daily dynamic workflow",
    statuses: { idle: "idle", running: "running", success: "success", failed: "failed" },
    stages: {
      fetch: "Fetch data",
      feature_build: "Build features",
      inference: "Run inference",
      strategy: "Select strategy",
      ranking: "Rank assets",
      export: "Export result",
    },
  },
  zh: {
    eyebrow: "Pipeline 监控",
    title: "每日动态流程",
    statuses: { idle: "待机", running: "运行中", success: "成功", failed: "失败" },
    stages: {
      fetch: "抓取市场数据",
      feature_build: "构建特征",
      inference: "模型推理",
      strategy: "选择策略",
      ranking: "资产排序",
      export: "导出结果",
    },
  },
};

const MESSAGE_ZH: Record<string, string> = {
  fetch: "已加载当前市场资产",
  feature_build: "已构建最新行情特征",
  inference: "已完成 FinVerse adapter 推理",
  strategy: "已选择匹配当前市场的策略",
  ranking: "已完成资产评分和排序",
  export: "已写入最新推荐结果",
};

export function PipelinePanel({ stages, language }: PipelinePanelProps) {
  const copy = COPY[language];
  return (
    <section className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-5">
      <p className="text-xs uppercase tracking-[0.25em] text-slate-500">{copy.eyebrow}</p>
      <h2 className="mt-2 text-xl font-semibold text-white">{copy.title}</h2>
      <div className="mt-5 space-y-3">
        {stages.map((stage) => (
          <div key={stage.name} className="flex items-center gap-3 rounded-2xl bg-white/[0.04] px-4 py-3">
            {icon(stage.status)}
            <div className="min-w-0 flex-1">
              <p className="font-medium text-slate-100">{copy.stages[stage.name as keyof typeof copy.stages] ?? stage.name}</p>
              <p className="truncate text-xs text-slate-500">{language === "zh" ? MESSAGE_ZH[stage.name] ?? stage.message ?? stage.status : stage.message || stage.status}</p>
            </div>
            <span className="rounded-full bg-slate-900 px-2 py-1 text-xs text-slate-400">{copy.statuses[stage.status] ?? stage.status}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
