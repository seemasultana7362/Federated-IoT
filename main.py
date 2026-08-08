import sys
from pathlib import Path
import yaml
import flwr as fl

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f) or {}

num_clients = int(config.get("num_clients", 5))
num_rounds = int(config.get("num_rounds", 3))

from client.client import FlowerClient
from server.strategy import FedAvgStrategy

strategy = FedAvgStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=num_clients,
    min_evaluate_clients=num_clients,
    min_available_clients=num_clients,
)

print("Flower setup complete. Start the server and clients separately if needed.")
print("Use: python -m flwr server --server-address 127.0.0.1:8080 --config num_rounds=3")
