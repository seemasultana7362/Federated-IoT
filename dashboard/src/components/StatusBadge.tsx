import type { ClientStatus, ExperimentStatus } from "../types"

const experimentClass: Record<ExperimentStatus, string> = {
  queued: "border-muted/40 text-muted",
  running: "border-accent/50 text-accent",
  completed: "border-ok/50 text-ok",
  failed: "border-danger/50 text-danger",
}

const clientClass: Record<ClientStatus, string> = {
  idle: "border-muted/40 text-muted",
  waiting: "border-warn/50 text-warn",
  training: "border-accent/50 text-accent",
  masked: "border-ok/50 text-ok",
  failed: "border-danger/50 text-danger",
}

export function ExperimentBadge({ status }: { status: ExperimentStatus }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide ${experimentClass[status]}`}
    >
      {status}
    </span>
  )
}

export function ClientBadge({ status }: { status: ClientStatus }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide ${clientClass[status]}`}
    >
      {status}
    </span>
  )
}

export function SourceBadge({ source }: { source: "demo" | "live" }) {
  const live = source === "live"
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide ${
        live ? "border-ok/50 text-ok" : "border-warn/50 text-warn"
      }`}
    >
      {live ? "Live Flower data" : "Demo data"}
    </span>
  )
}
