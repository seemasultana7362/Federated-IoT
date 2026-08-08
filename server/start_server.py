import os
from pathlib import Path

import yaml
import flwr as fl

from .strategy import FedAvgStrategy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f) or {}

num_clients = int(config.get("num_clients", 5))
num_rounds = int(config.get("num_rounds", 3))
server_address = os.environ.get("FLOWER_SERVER_ADDRESS", config.get("server_address", "127.0.0.1:9090"))

strategy = FedAvgStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=num_clients,
    min_evaluate_clients=num_clients,
    min_available_clients=num_clients,
)

fl.server.start_server(
    server_address=server_address,
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)
