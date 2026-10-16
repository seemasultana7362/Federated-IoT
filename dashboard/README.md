# Federated IoT 3D Experiment Console

An interactive React + Three.js control surface for the existing Python Flower / PyTorch / Opacus repository.

## Start locally

Use Python 3.11 in the project root and create/activate your project virtual environment first.

Terminal 1 — from the repository root:
```bash
pip install -r requirements.txt
python dashboard_api.py
```

Terminal 2:
```bash
cd dashboard
npm install
npm run dev
```

Open the Vite URL printed in Terminal 2 (usually http://localhost:5173). The dashboard calls the local API on http://127.0.0.1:8765.

## What Run does

- Validates the requested local client CSVs and config.
- Writes the selected Flower app configuration into a dedicated run record, launches the Flower simulation command in a Python child process, and streams logs/status to the dashboard.
- Tracks job status and reads real round metrics from `results/secure_aggregation/round_history.csv` when available.
- Does not invent model results. Global evaluation requires `dataset/processed/test_processed.csv`.

## Repository facts / integration notes

The active app defined in `pyproject.toml` is `server_app:app` and `client_app:app`; the server config uses hyphenated Flower run-config keys. The current global evaluator in `server/evaluate_global.py` assumes 42 features and a binary MLP. The app code currently hard-codes five clients and always creates Flower's SecAgg+ workflow, so this console rejects non-five-client runs and requires secure aggregation to be enabled for the active app path. Adaptive communication is displayed as an unsupported UI control until its implementation is wired into Flower.

The local API is intended for localhost use only. Do not expose it publicly without authentication and request controls.
