import { useEffect, useState } from "react"
import { fetchPrivacy } from "../api"
import { DataSourceBanner, ErrorState, LoadingState } from "../components/DataSourceBanner"
import type { PrivacyView } from "../types"

const STAGES = [
  { id: "setup", label: "0 Setup" },
  { id: "share_keys", label: "1 Share keys" },
  { id: "masked", label: "2 Masked vectors" },
  { id: "unmask", label: "3 Unmask" },
]

function Flag({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={ok ? "text-ok" : "text-warn"}>{label}</span>
  )
}

export function PrivacyPage() {
  const [privacy, setPrivacy] = useState<PrivacyView | null>(null)
  const [source, setSource] = useState<"demo" | "live">("demo")
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const next = await fetchPrivacy()
        if (!cancelled) {
          setPrivacy(next.privacy)
          setSource(next.source)
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load privacy state")
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

  if (loading) return <LoadingState label="Loading privacy state…" />
  if (error) return <ErrorState message={error} />
  if (!privacy) return null

  const stage = privacy.secagg_stage

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">Privacy & security</h2>
        <p className="mt-1 text-sm text-muted">
          Configured flags come from pyproject / run-config. Confirmed flags require matching Flower log or JSON events.
        </p>
      </div>

      <DataSourceBanner
        source={source}
        note={
          source === "demo"
            ? "This screen is a layout preview. Differential Privacy and SecAgg+ are not confirmed."
            : "Confirmed means the backend parsed matching runtime evidence. Configured-only means the flag was requested but not observed yet."
        }
      />

      <div className="grid gap-4 md:grid-cols-2">
        <section className="rounded-lg border border-line bg-panel p-4">
          <h3 className="font-medium">Differential Privacy</h3>
          <dl className="mt-3 space-y-2 text-sm">
            <div className="flex justify-between">
              <dt className="text-muted">Configured</dt>
              <dd>
                <Flag
                  ok={privacy.differential_privacy_configured}
                  label={privacy.differential_privacy_configured ? "yes" : "no"}
                />
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">Confirmed in logs</dt>
              <dd>
                <Flag
                  ok={privacy.differential_privacy_confirmed}
                  label={privacy.differential_privacy_confirmed ? "yes" : "not confirmed"}
                />
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">Target ε</dt>
              <dd>{privacy.target_epsilon ?? "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">Reported ε</dt>
              <dd>{privacy.reported_epsilon ?? "not reported"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">δ</dt>
              <dd>{privacy.target_delta ?? "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">Clipping C</dt>
              <dd>{privacy.max_grad_norm ?? "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">Noise multiplier</dt>
              <dd>{privacy.noise_multiplier ?? "not reported"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">Accountant</dt>
              <dd>{privacy.accountant}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">Optimizer</dt>
              <dd>{privacy.optimizer}</dd>
            </div>
          </dl>
        </section>

        <section className="rounded-lg border border-line bg-panel p-4">
          <h3 className="font-medium">SecAgg+</h3>
          <dl className="mt-3 space-y-2 text-sm">
            <div className="flex justify-between">
              <dt className="text-muted">Configured</dt>
              <dd>
                <Flag ok={privacy.secagg_configured} label={privacy.secagg_configured ? "yes" : "no"} />
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">Confirmed in logs</dt>
              <dd>
                <Flag
                  ok={privacy.secagg_confirmed}
                  label={privacy.secagg_confirmed ? "yes" : "not confirmed"}
                />
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">Shares</dt>
              <dd>{privacy.num_shares}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">Reconstruct threshold</dt>
              <dd>{privacy.reconstruction_threshold}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-muted">min_fit_clients</dt>
              <dd>{privacy.min_fit_clients}</dd>
            </div>
          </dl>
          <ol className="mt-4 grid grid-cols-2 gap-2">
            {STAGES.map((item) => {
              const active = stage === item.id
              return (
                <li
                  key={item.id}
                  className={`rounded-md border px-3 py-2 text-sm ${
                    active ? "border-accent text-accent" : "border-line text-muted"
                  }`}
                >
                  {item.label}
                </li>
              )
            })}
          </ol>
          <p className="mt-3 text-xs text-muted">
            Stage highlighting requires `[SECAGG+]` log lines from SecAggPlusWorkflowWithLogs (enabled for dashboard runs).
          </p>
        </section>
      </div>

      <section>
        <h3 className="text-lg font-medium">Warnings</h3>
        <ul className="mt-3 space-y-2">
          {privacy.warnings.map((warning) => (
            <li key={warning} className="rounded-md border border-warn/20 bg-warn/5 px-3 py-2 text-sm text-warn">
              {warning}
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
