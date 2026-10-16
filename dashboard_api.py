"""Local-only API for the Federated IoT 3D experiment dashboard.

This API deliberately binds to 127.0.0.1; do not expose it to the public internet.
It launches the repository's Flower app, streams logs, and reports real round metrics.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CLIENT_ROOT = ROOT / "dataset" / "clients"
TEST_PATH = ROOT / "dataset" / "processed" / "test_processed.csv"
ROUND_HISTORY = ROOT / "results" / "secure_aggregation" / "round_history.csv"
RUNS_DIR = ROOT / "dashboard" / ".runs"
MAX_UPLOAD = 250 * 1024 * 1024
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_preview(path: Path) -> tuple[bool, str, int | None]:
    """Return readiness, reason, and row preview count without reading a whole CSV."""
    if not path.exists():
        return False, f"Missing file: {path.relative_to(ROOT)}", None
    try:
        with path.open("rb") as handle:
            prefix = handle.read(160)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return False, (
                f"{path.relative_to(ROOT)} is a Git LFS pointer, not the dataset bytes. "
                "Run 'git lfs install' and 'git lfs pull' in the repository root."
            ), None
        if len(prefix.strip()) < 4:
            return False, f"{path.relative_to(ROOT)} is empty.", None
        import pandas as pd

        sample = pd.read_csv(path, nrows=5)
        if "label" not in sample.columns:
            return False, f"{path.relative_to(ROOT)} has no 'label' column.", None
        features = [c for c in sample.columns if c not in ("label", "attack_cat")]
        if len(features) != 42:
            return False, (
                f"{path.relative_to(ROOT)} has {len(features)} features; "
                "the current MLP/global evaluator requires exactly 42."
            ), None
        non_numeric = sample[features].select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            return False, (
                f"{path.relative_to(ROOT)} has non-numeric feature columns in its preview: "
                + ", ".join(non_numeric[:8])
            ), None
        return True, "ready", len(sample)
    except Exception as exc:
        return False, f"Could not validate {path.relative_to(ROOT)}: {exc}", None


def readiness(split: str = "non_iid", clients: int = 5) -> dict[str, Any]:
    split_dir = CLIENT_ROOT / split
    client_checks = []
    for client_id in range(1, clients + 1):
        path = split_dir / f"client_{client_id}.csv"
        ok, message, preview = read_preview(path)
        client_checks.append({
            "id": client_id, "ok": ok,
            "file": str(path.relative_to(ROOT)),
            "message": message, "preview_rows": preview,
        })
    test_ok, test_message, _ = read_preview(TEST_PATH)
    return {
        "split": split,
        "clients": client_checks,
        "ready_clients": sum(1 for c in client_checks if c["ok"]),
        "test_data": test_ok,
        "test_message": test_message,
        "ready": all(c["ok"] for c in client_checks) and test_ok,
    }


def load_metrics() -> list[dict[str, Any]]:
    if not ROUND_HISTORY.exists():
        return []
    try:
        with ROUND_HISTORY.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        parsed = []
        for row in rows:
            entry: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    entry[key] = float(value)
                except (ValueError, TypeError):
                    entry[key] = value
            parsed.append(entry)
        return parsed
    except Exception:
        return []


def parse_stage(line: str) -> int:
    lower = line.lower()
    # Flower evaluates round 0 before training. Treat that as preflight/baseline,
    # not as near-complete progress.
    if "global evaluation" in lower:
        if re.search(r"round\\s+0\\b", lower):
            return 1
        return 5
    if any(s in lower for s in ("masked vectors", "share keys", "unmask", "[secagg+] stage")):
        return 4
    if "[dp]" in lower or "privacy accounting" in lower:
        return 3
    if any(s in lower for s in ("training...", "local training", "epoch ", "fit response")):
        return 2
    if any(s in lower for s in ("loading client dataset", "preflight passed", "starting client")):
        return 1
    return 0


def append_log(job: dict[str, Any], line: str) -> None:
    with JOBS_LOCK:
        job["logs"].append(line.rstrip("\r\n"))
        if len(job["logs"]) > 800:
            job["logs"] = job["logs"][-800:]
        job["stage"] = max(job.get("stage", 0), parse_stage(line))
    lower = line.lower()
    if "epsilon" in lower:
        match = re.search(r"(?:ε|epsilon)[^0-9]{0,15}([0-9]+(?:\.[0-9]+)?)", line, re.I)
        if match:
            try:
                with JOBS_LOCK:
                    job["metrics"]["epsilon"] = float(match.group(1))
            except ValueError:
                pass


def refresh_job_metrics(job: dict[str, Any]) -> None:
    rows = load_metrics()
    start_index = job.get("history_start", 0)
    new_rows = rows[start_index:]
    if new_rows:
        latest = new_rows[-1]
        with JOBS_LOCK:
            for key in ("accuracy", "precision", "recall", "f1", "loss", "round"):
                if key in latest:
                    out_key = "f1" if key == "f1" else key
                    try:
                        job["metrics"][out_key] = float(latest[key])
                    except (ValueError, TypeError):
                        pass
            job["metrics"]["round_history"] = new_rows
            if float(latest.get("round", 0) or 0) >= 1:\n                job["stage"] = max(job.get("stage", 0), 5)


def run_flower(job_id: str, config: dict[str, Any]) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["started_at"] = utc_now()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["FEDERATED_IOT_SPLIT"] = config["split"]
    run_config = " ".join([
        f"num-server-rounds={config['rounds']}",
        f"local-epochs={config['epochs']}",
        f"learning-rate={config['lr']}",
        f"num-clients={config['clients']}",
        f"differential-privacy={'true' if config['dp'] else 'false'}",
        f"target-epsilon={config['epsilon']}",
        f"target-delta={config['delta']}",
        f"max-grad-norm={config['clip']}",
        f"num-shares={config['shares']}",
        f"reconstruction-threshold={config['threshold']}",
        f"secure-aggregation={'true' if config['secagg'] else 'false'}",
    ])
    command = [
        sys.executable, "-m", "flwr", "run", ".",
        "--federation-config", f"num-supernodes={config['clients']}",
        "--run-config", run_config,
    ]
    append_log(job, "[RUNNER] " + " ".join(command))
    append_log(job, f"[RUNNER] Dataset split: {config['split']}; clients={config['clients']}")
    append_log(job, "[RUNNER] Raw client CSV files remain local. Flower run starting…")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with JOBS_LOCK:
            job["pid"] = process.pid
        assert process.stdout is not None
        for line in process.stdout:
            append_log(job, line)
            refresh_job_metrics(job)
        exit_code = process.wait()
        refresh_job_metrics(job)
        with JOBS_LOCK:
            job["exit_code"] = exit_code
            job["finished_at"] = utc_now()
            if exit_code == 0:
                job["status"] = "succeeded"
                job["stage"] = 6
                job["logs"].append("[RUNNER] Flower process completed successfully.")
            else:
                job["status"] = "failed"
                job["error"] = (
                    f"Flower exited with code {exit_code}. See the execution stream for the "
                    "traceback; confirm the installed Flower version matches pyproject.toml."
                )
                job["logs"].append(f"[RUNNER] Flower exited with code {exit_code}.")
        record = {
            "job_id": job_id,
            "created_at": job["created_at"],
            "finished_at": job.get("finished_at"),
            "status": job["status"],
            "config": config,
            "metrics": job["metrics"],
            "exit_code": job.get("exit_code"),
            "error": job.get("error"),
        }
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        (RUNS_DIR / f"{job_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    except FileNotFoundError as exc:
        with JOBS_LOCK:
            job["status"] = "failed"
            job["error"] = (
                "Flower CLI not found in this Python environment. Install project requirements "
                "inside the active virtual environment, then restart dashboard_api.py."
            )
            job["logs"].append(f"[ERROR] {exc}")
            job["finished_at"] = utc_now()
    except Exception as exc:
        with JOBS_LOCK:
            job["status"] = "failed"
            job["error"] = str(exc)
            job["logs"].append(f"[ERROR] {type(exc).__name__}: {exc}")
            job["finished_at"] = utc_now()


def valid_config(data: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        config = {
            "clients": int(data.get("clients", 5)),
            "rounds": int(data.get("rounds", 3)),
            "epochs": int(data.get("epochs", 3)),
            "lr": float(data.get("lr", 0.001)),
            "split": str(data.get("split", "non_iid")),
            "dp": bool(data.get("dp", True)),
            "epsilon": float(data.get("epsilon", 2.0)),
            "delta": float(data.get("delta", 1e-5)),
            "clip": float(data.get("clip", 1.0)),
            "secagg": bool(data.get("secagg", True)),
            "shares": int(data.get("shares", 5)),
            "threshold": int(data.get("threshold", 3)),
            "adaptive": bool(data.get("adaptive", False)),
        }
    except (TypeError, ValueError) as exc:
        return None, f"Invalid configuration value: {exc}"
    if config["split"] not in ("iid", "non_iid"):
        return None, "Split must be IID or Non-IID."
    if not 2 <= config["clients"] <= 5:
        return None, "Select between 2 and 5 clients; the repository has five client partitions."
    if not 1 <= config["rounds"] <= 100:
        return None, "Global rounds must be between 1 and 100."
    if not 1 <= config["epochs"] <= 20:
        return None, "Local epochs must be between 1 and 20."
    if not 1e-5 <= config["lr"] <= 1:
        return None, "Learning rate must be in [0.00001, 1]."
    if config["dp"] and (config["epsilon"] <= 0 or not 0 < config["delta"] < 1 or config["clip"] <= 0):
        return None, "DP requires epsilon > 0, delta between 0 and 1, and a positive clipping norm."
    if config["secagg"]:
        if not 2 <= config["threshold"] <= config["shares"] <= config["clients"]:
            return None, "Secure aggregation requires 2 ≤ threshold ≤ shares ≤ selected clients."
    if config["adaptive"]:
        return None, "Adaptive communication is not wired into the active Flower workflow yet; disable it for this run."
    return config, None


class Handler(BaseHTTPRequestHandler):
    server_version = "FederatedIoTLocalAPI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[dashboard-api] " + fmt % args)

    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        origin = self.headers.get("Origin", "")
        allowed_origins = {
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:4173", "http://127.0.0.1:4173",
        }
        self.send_header("Access-Control-Allow-Origin", origin if origin in allowed_origins else "http://localhost:5173")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        origin = self.headers.get("Origin", "")
        allowed_origins = {
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:4173", "http://127.0.0.1:4173",
        }
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin if origin in allowed_origins else "http://localhost:5173")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            status = readiness()
            self._send(200, {
                "repo": True,
                "clients": status["ready_clients"],
                "test_data": status["test_data"],
                "ready": status["ready"],
                "message": "Ready" if status["ready"] else "Preflight needs attention",
                "details": status,
            })
            return
        if self.path == "/api/history":
            self._send(200, {"items": load_metrics()[-500:]})
            return
        if self.path.startswith("/api/jobs/"):
            job_id = self.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    self._send(404, {"error": "Job not found"})
                    return
                snapshot = {k: v for k, v in job.items() if k not in ("process",)}
            self._send(200, snapshot)
            return
        self._send(404, {"error": "Endpoint not found"})

    def do_POST(self) -> None:
        if self.path == "/api/upload-test":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_UPLOAD:
                    self._send(413, {"error": "Upload must be between 1 byte and 250 MB."})
                    return
                filename = self.headers.get("X-Filename", "test_processed.csv")
                if not filename.lower().endswith(".csv"):
                    self._send(400, {"error": "Upload a processed CSV file."})
                    return
                content = self.rfile.read(length)
                if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
                    self._send(400, {"error": "This is a Git LFS pointer, not a CSV dataset. Pull the actual LFS object first."})
                    return
                import pandas as pd
                import io
                preview = pd.read_csv(io.BytesIO(content), nrows=5)
                if "label" not in preview.columns:
                    self._send(400, {"error": "CSV must include a 'label' target column."})
                    return
                feature_cols = [c for c in preview.columns if c not in ("label", "attack_cat")]
                if len(feature_cols) != 42:
                    self._send(400, {"error": f"CSV has {len(feature_cols)} model features. This repository's global evaluator requires exactly 42 processed numeric features; run the project's preprocessing pipeline first."})
                    return
                if preview[feature_cols].select_dtypes(exclude=["number"]).shape[1]:
                    self._send(400, {"error": "Feature columns must already be numeric and preprocessed."})
                    return
                TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
                temp = TEST_PATH.with_suffix(".csv.uploading")
                temp.write_bytes(content)
                temp.replace(TEST_PATH)
                self._send(200, {"success": True, "message": "Processed held-out test CSV saved to dataset/processed/test_processed.csv"})
            except Exception as exc:
                self._send(400, {"error": f"Could not accept test CSV: {exc}"})
            return
        if self.path != "/api/run":
            self._send(404, {"error": "Endpoint not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                self._send(413, {"error": "Invalid request body."})
                return
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            config, problem = valid_config(data)
            if problem or config is None:
                self._send(400, {"error": problem or "Invalid config"})
                return
            checks = readiness(config["split"], config["clients"])
            failures = [f"Client {item['id']}: {item['message']}" for item in checks["clients"] if not item["ok"]]
            if not checks["test_data"]:
                failures.append("Global test set: " + checks["test_message"])
            if failures:
                self._send(400, {
                    "error": "Preflight failed. Fix the dataset issue(s) before launching training.",
                    "details": failures,
                    "preflight": checks,
                })
                return
            job_id = uuid.uuid4().hex
            job = {
                "job_id": job_id,
                "created_at": utc_now(),
                "status": "running",
                "stage": 0,
                "logs": ["[PREFLIGHT] Dataset files and 42-feature schema validated."],
                "metrics": {"accuracy": None, "precision": None, "recall": None, "f1": None, "loss": None, "epsilon": None},
                "config": config,
                "history_start": len(load_metrics()),
                "error": None,
            }
            with JOBS_LOCK:
                JOBS[job_id] = job
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            (RUNS_DIR / f"{job_id}.json").write_text(json.dumps({
                "job_id": job_id, "created_at": job["created_at"], "status": "running", "config": config
            }, indent=2), encoding="utf-8")
            thread = threading.Thread(target=run_flower, args=(job_id, config), daemon=True)
            thread.start()
            self._send(202, {"job_id": job_id, "status": "running", "logs": job["logs"]})
        except json.JSONDecodeError:
            self._send(400, {"error": "Request body must be valid JSON."})
        except Exception as exc:
            self._send(500, {"error": f"Could not start experiment: {exc}"})


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Federated IoT dashboard API listening on http://127.0.0.1:8765")
    print("Localhost only. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard API…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
