export type ExperimentStatus = "queued" | "running" | "completed" | "failed"

export type DataSource = "demo" | "live"

export type ClientStatus = "idle" | "training" | "masked" | "waiting" | "failed"

export interface RoundMetrics {
  round: number
  loss: number
  accuracy: number
  precision: number
  recall: number
  f1: number
  confusion_matrix?: number[][]
}

export interface ClientView {
  id: number
  name: string
  samples: number
  status: ClientStatus
  last_update: string
  local_accuracy?: number | null
  epsilon?: number | null
}

export interface PrivacyView {
  differential_privacy_configured: boolean
  differential_privacy_confirmed: boolean
  secagg_configured: boolean
  secagg_confirmed: boolean
  target_epsilon: number | null
  reported_epsilon: number | null
  target_delta: number | null
  max_grad_norm: number | null
  noise_multiplier: number | null
  accountant: string
  optimizer: string
  num_shares: number
  reconstruction_threshold: number
  min_fit_clients: number
  secagg_stage: string | null
  warnings: string[]
}

export interface ExperimentConfig {
  num_clients: number
  num_rounds: number
  local_epochs: number
  learning_rate: number
  differential_privacy: boolean
  target_epsilon: number
  target_delta: number
  max_grad_norm: number
  num_shares: number
  reconstruction_threshold: number
}

export interface ExperimentSummary {
  id: string
  status: ExperimentStatus
  created_at: string
  updated_at: string
  config: ExperimentConfig
  error?: string | null
  current_round?: number | null
  source?: DataSource
}

export interface ExperimentDetail extends ExperimentSummary {
  metrics: RoundMetrics[]
  clients: ClientView[]
  privacy: PrivacyView
  logs_tail: string[]
}

export interface OverviewData {
  source: DataSource
  source_note: string
  experiment: ExperimentSummary | null
  latest_metrics: RoundMetrics | null
  clients: ClientView[]
  privacy: PrivacyView
  limitations: string[]
}

export interface ApiHealth {
  status: string
  project_root: string
  flower_cli: boolean
  concurrent_run: boolean
}
