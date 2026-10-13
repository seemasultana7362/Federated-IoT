import { useEffect, useState } from "react"
import { fetchOverview } from "../api"
import { DataSourceBanner, ErrorState, LoadingState } from "../components/DataSourceBanner"
import { ExperimentBadge } from "../components/StatusBadge"
import type { OverviewData } from "../types"

function MetricCard({
  label,
  value,
  hint,
  placeholder,
}: {
  label: string
  value: string
  hint: string
  placeholder?: boolean
}) {
  return (
    <div className="rounded-lg border border-line bg-panel p-4">
      <p className="text-xs uppercase tracking-wide text-muted">{label}</p>
      <p className={`mt-2 text-2xl font-semibold ${placeholder ? "text-muted" : "text-ink"}`}>
        {value}
      </p>
      <p className="mt-1 text-xs text-muted">{hint}</p>
    </div>
  )
}

export function OverviewPage() {
  const [data, setData] = useState<OverviewData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const next = await fetchOverview()
        if (!cancelled) setData(next)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load overview")
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void load()
    const timer = window.setInterval(() => void load(), 4000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  if (loading && !data) return <LoadingState label="Loading overview…" />
  if (error) return <ErrorState message={error} />
  if (!data) return null

  const metrics = data.latest_metrics
  const experiment = data.experiment
  const missingMetrics = data.source === "demo" || !metrics

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">Overview</h2>
        <p className="mt-1 text-sm text-muted">
          Runtime snapshot of the last Flower experiment. Empty metrics mean evaluation has not written a round yet.
        </p>
      </div>

      <DataSourceBanner source={data.source} note={data.source_note} />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Experiment"
          value={experiment ? experiment.status : "none"}
          hint={experiment ? experiment.id : "Start a run from Experiments"}
        />
        <MetricCard
          label="Active clients"
          value={`${data.clients.length} / 5`}
          hint="Partitions under dataset/clients/non_iid"
        />
        <MetricCard
          label="Current round"
          value={experiment?.current_round != null ? String(experiment.current_round) : "—"}
          hint={experiment ? `${experiment.config.num_rounds} configured` : "No run"}
          placeholder={experiment?.current_round == null}
        />
        <MetricCard
          label="DP / SecAgg+"
          value={`${data.privacy.differential_privacy_confirmed ? "DP confirmed" : "DP unconfirmed"}`}
          hint={
            data.privacy.secagg_confirmed
              ? "SecAgg+ confirmed in logs"
              : "SecAgg+ not confirmed — configured only"
          }
          placeholder={!data.privacy.differential_privacy_confirmed}
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <MetricCard
          label="Global accuracy"
          value={metrics ? `${metrics.accuracy.toFixed(2)}%` : "no live metric"}
          hint="From server/evaluate_global.py on held-out test CSV"
          placeholder={missingMetrics}
        />
        <MetricCard
          label="Global F1"
          value={metrics ? `${metrics.f1.toFixed(2)}%` : "no live metric"}
          hint="Weighted F1, zero_division=0"
          placeholder={missingMetrics}
        />
        <MetricCard
          label="Global loss"
          value={metrics ? metrics.loss.toFixed(4) : "no live metric"}
          hint="Cross-entropy on test_processed.csv"
          placeholder={missingMetrics}
        />
      </div>

      {experiment && (
        <div className="rounded-lg border border-line bg-panel p-4">
          <div className="flex flex-wrap items-center gap-3">
            <ExperimentBadge status={experiment.status} />
            <span className="text-sm text-muted">Rounds {experiment.config.num_rounds}</span>
            <span className="text-sm text-muted">Epochs {experiment.config.local_epochs}</span>
            <span className="text-sm text-muted">lr {experiment.config.learning_rate}</span>
            <span className="text-sm text-muted">
              target ε {experiment.config.target_epsilon}
            </span>
          </div>
        </div>
      )}

      <section>
        <h3 className="text-lg font-medium">Implementation warnings</h3>
        <ul className="mt-3 space-y-2">
          {data.limitations.map((item) => (
            <li key={item} className="rounded-md border border-warn/20 bg-warn/5 px-3 py-2 text-sm text-warn">
              {item}
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
