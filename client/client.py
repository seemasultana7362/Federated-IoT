import argparse
from collections import OrderedDict
from xml.parsers.expat import model

import flwr as fl
import torch
from torch import nn

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
        target_epsilon = float(config.get("target_epsilon", 2.0))
        target_delta = float(config.get("target_delta", 1e-5))
        max_grad_norm = float(config.get("max_grad_norm", 1.0))

        (
            self.model,
            losses,
            accuracies,
            privacy_metrics,
        ) = train_model(
            model=self.model,
            train_loader=self.train_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=self.device,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
        )

        print(
            f"Client {self.cid}: "
            f"Training Finished."
        )

        print(
            f"Client {self.cid}: "
            f"ε = {privacy_metrics['epsilon']:.4f}, "
            f"δ = {privacy_metrics['delta']:.2e}"
        )

        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {
                "epsilon": privacy_metrics["epsilon"],
                "delta": privacy_metrics["delta"],
            },
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