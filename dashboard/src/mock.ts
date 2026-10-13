import type {
  ClientView,
  ExperimentDetail,
  OverviewData,
  PrivacyView,
  RoundMetrics,
} from "./types"

export const DEMO_LIMITATIONS = [
  "Privacy budget is measured per round. A new Opacus PrivacyEngine is created on every fit(), so displayed ε is not composed across federated rounds.",
  "Local optimizer is Adam wrapped by Opacus (DP-Adam), not classical DP-SGD.",
  "FedAvg sets min_fit_clients=5, so SecAgg+’s reconstruction threshold of 3 cannot actually tolerate dropouts.",
  "Client.evaluate() still scores the local training set; paper metrics come only from server/evaluate_global.py on the held-out test CSV.",
  "aggregation/fedavg.py, aggregation/secure_aggregation.py, privacy/dp_config.py and communication/adaptive.py are unused stubs.",
]

export const DEMO_PRIVACY: PrivacyView = {
  differential_privacy_configured: true,
  differential_privacy_confirmed: false,
  secagg_configured: true,
  secagg_confirmed: false,
  target_epsilon: 2.0,
  reported_epsilon: null,
  target_delta: 1e-5,
  max_grad_norm: 1.0,
  noise_multiplier: null,
  accountant: "prv (configured)",
  optimizer: "Adam / DP-Adam",
  num_shares: 5,
  reconstruction_threshold: 3,
  min_fit_clients: 5,
  secagg_stage: null,
  warnings: [
    "DEMO DATA: DP and SecAgg+ are shown as configured, not confirmed. Confirmation requires a live Flower process.",
    ...DEMO_LIMITATIONS,
  ],
}

export const DEMO_CLIENTS: ClientView[] = [
  {
    id: 1,
    name: "Client 1",
    samples: 66491,
    status: "idle",
    last_update: "demo",
    local_accuracy: null,
    epsilon: null,
  },
  {
    id: 2,
    name: "Client 2",
    samples: 42000,
    status: "idle",
    last_update: "demo",
    local_accuracy: null,
    epsilon: null,
  },
  {
    id: 3,
    name: "Client 3",
    samples: 35139,
    status: "idle",
    last_update: "demo",
    local_accuracy: null,
    epsilon: null,
  },
  {
    id: 4,
    name: "Client 4",
    samples: 19317,
    status: "idle",
    last_update: "demo",
    local_accuracy: null,
    epsilon: null,
  },
  {
    id: 5,
    name: "Client 5",
    samples: 12394,
    status: "idle",
    last_update: "demo",
    local_accuracy: null,
    epsilon: null,
  },
]

export const DEMO_ROUNDS: RoundMetrics[] = [
  {
    round: 0,
    loss: 0.62,
    accuracy: 68.4,
    precision: 67.1,
    recall: 68.4,
    f1: 66.9,
    confusion_matrix: [
      [1200, 800],
      [450, 2550],
    ],
  },
  {
    round: 1,
    loss: 0.41,
    accuracy: 81.2,
    precision: 80.4,
    recall: 81.2,
    f1: 80.7,
    confusion_matrix: [
      [1500, 500],
      [280, 2720],
    ],
  },
]

export const DEMO_OVERVIEW: OverviewData = {
  source: "demo",
  source_note:
    "UI layout only. Sample sizes match dataset/clients/non_iid CSV row counts. Accuracy, F1, loss, ε, and SecAgg+ stages are synthetic placeholders — not Flower results.",
  experiment: {
    id: "demo-preview",
    status: "queued",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    current_round: null,
    source: "demo",
    config: {
      num_clients: 5,
      num_rounds: 1,
      local_epochs: 3,
      learning_rate: 0.001,
      differential_privacy: true,
      target_epsilon: 2.0,
      target_delta: 1e-5,
      max_grad_norm: 1.0,
      num_shares: 5,
      reconstruction_threshold: 3,
    },
  },
  latest_metrics: null,
  clients: DEMO_CLIENTS,
  privacy: DEMO_PRIVACY,
  limitations: DEMO_LIMITATIONS,
}

export const DEMO_EXPERIMENT: ExperimentDetail = {
  ...DEMO_OVERVIEW.experiment!,
  metrics: [],
  clients: DEMO_CLIENTS,
  privacy: DEMO_PRIVACY,
  logs_tail: [
    "[DEMO] FastAPI is not connected.",
    "[DEMO] Start dashboard-api, then this page will show flwr run output.",
    "[DEMO] Do not treat the placeholder charts as experiment results.",
  ],
}

export const DEMO_CHART_ROUNDS: RoundMetrics[] = DEMO_ROUNDS
