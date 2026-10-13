import { DEMO_CLIENTS, DEMO_EXPERIMENT, DEMO_OVERVIEW, DEMO_PRIVACY } from "./mock"
import type {
  ClientView,
  ExperimentDetail,
  ExperimentSummary,
  OverviewData,
  PrivacyView,
  RoundMetrics,
} from "./types"

const API_BASE = "/api"

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  })
  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      /* ignore */
    }
    throw new ApiError(detail, response.status)
  }
  return (await response.json()) as T
}

export async function fetchHealth(): Promise<boolean> {
  try {
    const data = await request<{ status: string }>("/health")
    return data.status === "ok"
  } catch {
    return false
  }
}

export async function fetchOverview(): Promise<OverviewData> {
  try {
    return await request<OverviewData>("/overview")
  } catch {
    return DEMO_OVERVIEW
  }
}

export async function fetchClients(): Promise<{ source: "demo" | "live"; clients: ClientView[] }> {
  try {
    return await request("/clients")
  } catch {
    return { source: "demo", clients: DEMO_CLIENTS }
  }
}

export async function fetchPrivacy(): Promise<{ source: "demo" | "live"; privacy: PrivacyView }> {
  try {
    return await request("/privacy")
  } catch {
    return { source: "demo", privacy: DEMO_PRIVACY }
  }
}

export async function fetchExperiments(): Promise<ExperimentSummary[]> {
  try {
    return await request("/experiments")
  } catch {
    return [DEMO_EXPERIMENT]
  }
}

export async function fetchExperiment(id: string): Promise<ExperimentDetail> {
  try {
    return await request(`/experiments/${id}`)
  } catch {
    return DEMO_EXPERIMENT
  }
}

export async function fetchLogs(id: string): Promise<{ lines: string[] }> {
  try {
    return await request(`/experiments/${id}/logs`)
  } catch {
    return { lines: DEMO_EXPERIMENT.logs_tail }
  }
}

export async function fetchMetrics(id: string): Promise<{ source: "demo" | "live"; metrics: RoundMetrics[] }> {
  try {
    return await request(`/experiments/${id}/metrics`)
  } catch {
    return { source: "demo", metrics: [] }
  }
}

export async function startExperiment(body: Record<string, unknown>): Promise<ExperimentSummary> {
  return request("/experiments", {
    method: "POST",
    body: JSON.stringify(body),
  })
}
