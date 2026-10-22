import os
from pathlib import Path

import yaml
import flwr as fl

from .strategy import FedAvgStrategy


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# ============================================================
# Load Configuration
# ============================================================


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


# ============================================================
# Configuration
# ============================================================

config = load_config()
num_clients = int(config.get("num_clients", 5))
num_rounds = int(config.get("num_rounds", 20))
server_address = os.environ.get(
    "FLOWER_SERVER_ADDRESS",
    config.get("server_address", "127.0.0.1:9090"),
)


# ============================================================
# Federated Strategy
# ============================================================

strategy = FedAvgStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=num_clients,
    min_evaluate_clients=num_clients,
    min_available_clients=num_clients,
)


# ============================================================
# Start Flower Server
# ============================================================


def main() -> None:
    print("\n==============================================")
    print("      FEDERATED LEARNING SERVER")
    print("==============================================")
    print(f"Clients : {num_clients}")
    print(f"Rounds  : {num_rounds}")
    print(f"Address : {server_address}")
    print("Global evaluation: ENABLED")
    print("Automatic CSV logging: ENABLED")
    print("==============================================\n")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()