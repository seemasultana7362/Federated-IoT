import { useEffect, useMemo, useState } from "react"
import { fetchClients } from "../api"
import { DataSourceBanner, EmptyState, ErrorState, LoadingState } from "../components/DataSourceBanner"
import { ClientBadge, SourceBadge } from "../components/StatusBadge"
import type { ClientView } from "../types"

export function ClientsPage() {
  const [clients, setClients] = useState<ClientView[]>([])
  const [source, setSource] = useState<"demo" | "live">("demo")
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const next = await fetchClients()
        if (!cancelled) {
          setClients(next.clients)
          setSource(next.source)
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load clients")
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

  const total = useMemo(() => clients.reduce((sum, client) => sum + client.samples, 0), [clients])
  const max = Math.max(...clients.map((client) => client.samples), 1)

  if (loading) return <LoadingState label="Loading client partitions…" />
  if (error) return <ErrorState message={error} />
  if (clients.length === 0) {
    return (
      <EmptyState
        title="No client partitions found"
        body="Expected dataset/clients/non_iid/client_1.csv through client_5.csv."
      />
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">Federated clients</h2>
        <p className="mt-1 text-sm text-muted">
          Five non-IID partitions. Sample counts are file row counts. Training/update status comes from Flower logs when a run is live.
        </p>
      </div>

      <DataSourceBanner
        source={source}
        note={
          source === "demo"
            ? "Sample sizes below match the CSVs on disk, but training status is idle placeholder data until the API is connected."
            : "Sample sizes are live CSV row counts. Status is inferred from the active or latest experiment logs — not a Flower client RPC."
        }
      />

      <div className="rounded-lg border border-line bg-panel p-4">
        <p className="text-xs uppercase tracking-wide text-muted">Non-IID network</p>
        <div className="mt-4 flex flex-wrap items-end justify-center gap-4">
          <div className="flex h-24 w-24 items-center justify-center rounded-full border border-accent/40 text-center text-xs text-accent">
            Flower
            <br />
            server
          </div>
          {clients.map((client) => (
            <div key={client.id} className="w-28">
              <div
                className="rounded-md border border-line bg-panel-2 p-3 text-center"
                style={{ height: 72 + (client.samples / max) * 48 }}
              >
                <p className="text-sm font-medium">{client.name}</p>
                <p className="text-xs text-muted">{client.samples.toLocaleString()} rows</p>
              </div>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-muted">Bar height encodes relative partition size. Total {total.toLocaleString()} training rows.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {clients.map((client) => (
          <article key={client.id} className="rounded-lg border border-line bg-panel p-4">
            <div className="flex items-center justify-between">
              <h3 className="font-medium">{client.name}</h3>
              <ClientBadge status={client.status} />
            </div>
            <dl className="mt-3 space-y-1 text-sm">
              <div className="flex justify-between gap-4">
                <dt className="text-muted">Local samples</dt>
                <dd>{client.samples.toLocaleString()}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-muted">Last update</dt>
                <dd>{client.last_update}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-muted">Local accuracy</dt>
                <dd>
                  {client.local_accuracy == null
                    ? "not reported"
                    : `${client.local_accuracy.toFixed(2)}% (train set)`}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-muted">Reported ε</dt>
                <dd>{client.epsilon == null ? "not confirmed" : client.epsilon.toFixed(4)}</dd>
              </div>
            </dl>
            <p className="mt-3 text-xs text-muted">
              Local accuracy is training-set accuracy from FlowerClient.fit, not the held-out test metric.
            </p>
          </article>
        ))}
      </div>
      <SourceBadge source={source} />
    </div>
  )
}
