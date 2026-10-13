import { SourceBadge } from "./StatusBadge"

export function DataSourceBanner({
  source,
  note,
}: {
  source: "demo" | "live"
  note: string
}) {
  const demo = source === "demo"
  return (
    <div
      className={`rounded-lg border px-4 py-3 text-sm ${
        demo
          ? "border-warn/30 bg-warn/10 text-warn"
          : "border-ok/30 bg-ok/10 text-ink"
      }`}
    >
      <div className="mb-1 flex items-center gap-2">
        <SourceBadge source={source} />
        <span className="text-xs uppercase tracking-wide text-muted">
          Metric provenance
        </span>
      </div>
      <p className={demo ? "text-warn" : "text-ink"}>{note}</p>
    </div>
  )
}

export function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-lg border border-dashed border-line px-4 py-10 text-center">
      <p className="text-sm font-medium text-ink">{title}</p>
      <p className="mt-1 text-sm text-muted">{body}</p>
    </div>
  )
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">
      {message}
    </div>
  )
}

export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="rounded-lg border border-line bg-panel px-4 py-8 text-center text-sm text-muted">
      {label}
    </div>
  )
}
