import type { AssetDetail, AssetRecommendation, LiveQuote, LiveQuotesResponse, Market, PipelineStatus, RecommendationResponse } from "@/types";

type StaticAsset = AssetRecommendation & Pick<AssetDetail, "history_close" | "rollout_path">;
type StaticRecommendation = Omit<RecommendationResponse, "all_assets" | "top_assets"> & {
  top_assets: StaticAsset[];
  all_assets?: StaticAsset[];
  pipeline_status?: { stages?: PipelineStatus["stages"] };
  model_checkpoint?: string;
};

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${url} failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function withMarket(path: string, market: Market) {
  return `${path}?market=${encodeURIComponent(market)}`;
}

function staticDataUrl(market: Market) {
  return `${import.meta.env.BASE_URL}data/${market}/latest.json?v=${Date.now()}`;
}

const liveApiBase = ((import.meta.env.VITE_LIVE_API_BASE as string | undefined) ?? "").replace(/\/$/, "");

function liveApiUrl(path: string) {
  return `${liveApiBase}${path}`;
}

async function getStaticLatest(market: Market) {
  return getJson<StaticRecommendation>(staticDataUrl(market));
}

function quoteFromAsset(asset: StaticAsset | AssetRecommendation, market: Market, asOf: string): LiveQuote {
  const history = (asset as StaticAsset).history_close ?? [];
  const previousClose = history.length >= 2 ? history[history.length - 2].close : null;
  const change = previousClose ? asset.close - previousClose : null;
  const changePercent = previousClose ? asset.close / previousClose - 1 : null;
  return {
    market,
    ticker: asset.ticker,
    name: asset.name,
    price: asset.close,
    previous_close: previousClose,
    open: null,
    high: null,
    low: null,
    volume: null,
    change,
    change_percent: changePercent,
    currency: market === "cn" ? "CNY" : "USD",
    market_state: "SNAPSHOT",
    quote_time: asOf,
    source: "daily_snapshot_fallback",
    is_realtime: false,
  };
}

async function getStaticLiveQuotes(market: Market, tickers: string[] = []): Promise<LiveQuotesResponse> {
  const payload = await getStaticLatest(market);
  const allAssets = [...(payload.all_assets ?? []), ...payload.top_assets];
  const byTicker = new Map(allAssets.map((asset) => [asset.ticker.toUpperCase(), asset]));
  const selected = tickers.length
    ? tickers.map((ticker) => byTicker.get(ticker.toUpperCase())).filter((asset): asset is StaticAsset => Boolean(asset))
    : payload.top_assets;
  return {
    market,
    as_of: payload.last_updated_at,
    source: "daily_snapshot_fallback",
    is_realtime: false,
    quotes: selected.map((asset) => quoteFromAsset(asset, market, payload.last_updated_at)),
  };
}

async function getWithStaticFallback<T>(apiPath: string, market: Market, fallback: () => Promise<T>) {
  if (import.meta.env.VITE_DATA_MODE === "static") {
    return fallback();
  }
  try {
    return await getJson<T>(withMarket(apiPath, market));
  } catch {
    return fallback();
  }
}

export function getLatestRecommendation(market: Market) {
  return getWithStaticFallback<RecommendationResponse>("/api/recommendations/latest", market, async () => getStaticLatest(market));
}

export async function getLiveQuotes(market: Market, tickers: string[] = []) {
  const params = new URLSearchParams({ market });
  if (tickers.length) {
    params.set("symbols", tickers.join(","));
  }
  if (import.meta.env.VITE_DATA_MODE === "static" && !liveApiBase) {
    return getStaticLiveQuotes(market, tickers);
  }
  try {
    return await getJson<LiveQuotesResponse>(liveApiUrl(`/api/quotes/live?${params.toString()}`));
  } catch {
    return getStaticLiveQuotes(market, tickers);
  }
}

export function getPipelineStatus(market: Market) {
  return getWithStaticFallback<PipelineStatus>("/api/pipeline/status", market, async () => {
    const payload = await getStaticLatest(market);
    return {
      market: payload.market,
      language: payload.language,
      trade_date: payload.trade_date,
      last_updated_at: payload.last_updated_at,
      stages: payload.pipeline_status?.stages ?? [],
      model_checkpoint: payload.model_checkpoint ?? "",
      mode: payload.mode,
    };
  });
}

export function getAssetDetail(ticker: string, market: Market) {
  return getWithStaticFallback<AssetDetail>(`/api/assets/${ticker}`, market, async () => {
    const payload = await getStaticLatest(market);
    const asset = payload.all_assets?.find((item) => item.ticker.toUpperCase() === ticker.toUpperCase());
    if (!asset) {
      throw new Error(`asset not found in static demo: ${ticker}`);
    }
    return {
      ticker: asset.ticker,
      trade_date: payload.trade_date,
      history_close: asset.history_close,
      rollout_path: asset.rollout_path,
      features: {
        expected_return_30d: asset.expected_return_30d,
        predicted_volatility: asset.predicted_volatility,
        predicted_downside: asset.predicted_downside,
        bull_prob: asset.regime_probs.bull,
        sideway_prob: asset.regime_probs.sideway,
        bear_prob: asset.regime_probs.bear,
      },
      explanation: asset.reasons,
      score: asset.score,
      rank: asset.rank,
      sector: asset.sector,
      type: asset.type,
    };
  });
}

export async function runPipeline(market: Market) {
  if (import.meta.env.VITE_DATA_MODE === "static") {
    return { run_id: `static-demo-${market}`, status: "static-demo" };
  }
  const response = await fetch("/api/pipeline/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ market, force_fetch: false }),
  });
  if (response.ok) {
    return response.json() as Promise<{ run_id: string; status: string }>;
  }
  return { run_id: `static-demo-${market}`, status: "static-demo" };
}
