from flwr.client import ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from client.client import FlowerClient


def client_fn(context: Context):
    print(
        f"[DIAG][client_app] client_fn entered; "
        f"node_config={context.node_config}",
        flush=True,
    )
    partition_id = int(context.node_config["partition-id"])
    client_id = partition_id + 1

    print(
        f"[DIAG][client_app] creating client {client_id}",
        flush=True,
    )

    client = FlowerClient(cid=str(client_id))

    print(
        f"[DIAG][client_app] wrapping client {client_id} as Flower client",
        flush=True,
    )
    flower_client = client.to_client()
    print(
        f"[DIAG][client_app] client {client_id} ready",
        flush=True,
    )
    return flower_client


app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod],
)