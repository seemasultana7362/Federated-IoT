from flwr.client import ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from client.client import FlowerClient


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    client_id = partition_id + 1

    print(f"[CLIENT APP] Starting client {client_id}")

    client = FlowerClient(cid=str(client_id))

    return client.to_client()


app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod],
)