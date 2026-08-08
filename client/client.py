import argparse
from collections import OrderedDict

import flwr as fl
import torch

from client.dataset import load_client_dataset
from client.train import (
    build_client_model,
    train_model,
)

from client.evaluate import evaluate_model


# ============================================================
# Device
# ============================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Flower Client
# ============================================================

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, cid: str):
        self.cid = cid
        self.device = get_device()

        print(f"\n========== CLIENT {cid} ==========")

        # Load local dataset
        self.train_loader, self.input_size = load_client_dataset(int(cid))

        # Build model
        self.model = build_client_model(
            self.input_size,
            self.device,
        )

        print(
            f"Loaded Client {cid} "
            f"with {len(self.train_loader.dataset)} samples."
        )

    # --------------------------------------------------------

    def get_parameters(self, config):
        return [
            val.detach().cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    # --------------------------------------------------------

    def set_parameters(self, parameters):

        params_dict = zip(
            self.model.state_dict().keys(),
            parameters,
        )

        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in params_dict
            }
        )

        self.model.load_state_dict(state_dict, strict=True)

    # --------------------------------------------------------

    def fit(self, parameters, config):

        print(f"\nClient {self.cid}: Training...")

        self.set_parameters(parameters)

        epochs = int(config.get("epochs", 3))
        learning_rate = float(config.get("learning_rate", 0.001))

        self.model, losses, accuracies = train_model(
            model=self.model,
            train_loader=self.train_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=self.device,
        )

        print(
            f"Client {self.cid}: "
            f"Training Finished."
        )

        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {},
        )

    # --------------------------------------------------------

    def evaluate(self, parameters, config):

        print(f"\nClient {self.cid}: Evaluating...")

        self.set_parameters(parameters)

        metrics = evaluate_model(
            model=self.model,
            dataloader=self.train_loader,
            device=self.device,
        )

        print(
            f"Client {self.cid} | "
            f"Accuracy: {metrics['accuracy']:.2f}% | "
            f"Loss: {metrics['loss']:.4f}"
        )

        return (
            metrics["loss"],
            len(self.train_loader.dataset),
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            },
        )

# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        help="Client ID",
    )

    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:9090",
        help="Flower server address",
    )

    args = parser.parse_args()

    client = FlowerClient(str(args.client_id))

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()