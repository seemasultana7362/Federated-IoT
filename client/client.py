import argparse
from collections import OrderedDict

import flwr as fl
import torch

from client.dataset import (
    load_client_dataset,
)

from client.train import (
    build_client_model,
    train_model,
)

from client.evaluate import (
    evaluate_model,
)


# ============================================================
# Device
# ============================================================

def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# Flower Client
# ============================================================

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, cid: str):

        self.cid = cid

        self.device = get_device()

        print(
            f"[DIAG][client {cid}] initialization started; "
            f"device={self.device}",
            flush=True,
        )

        # ----------------------------------------------------
        # Load local dataset
        # ----------------------------------------------------

        print(
            f"[DIAG][client {cid}] loading dataset",
            flush=True,
        )
        self.train_loader, self.input_size = (
            load_client_dataset(
                int(cid)
            )
        )
        print(
            f"[DIAG][client {cid}] dataset loaded; "
            f"samples={len(self.train_loader.dataset)}, "
            f"features={self.input_size}",
            flush=True,
        )

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        print(
            f"[DIAG][client {cid}] building model",
            flush=True,
        )
        self.model = build_client_model(
            self.input_size,
            self.device,
        )

        print(
            f"[DIAG][client {cid}] initialization complete",
            flush=True,
        )

    # ========================================================
    # Get Parameters
    # ========================================================

    def get_parameters(self, config):

        return [
            val.detach()
            .cpu()
            .numpy()

            for _, val in (
                self.model.state_dict().items()
            )
        ]

    # ========================================================
    # Set Parameters
    # ========================================================

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

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

    # ========================================================
    # Federated Training
    # ========================================================

    def fit(
        self,
        parameters,
        config,
    ):

        print(
            f"[DIAG][client {self.cid}] fit started; "
            f"received {len(parameters)} parameter arrays",
            flush=True,
        )

        # ----------------------------------------------------
        # Load global model
        # ----------------------------------------------------

        self.set_parameters(
            parameters
        )
        print(
            f"[DIAG][client {self.cid}] global parameters loaded",
            flush=True,
        )

        # ----------------------------------------------------
        # Read training configuration
        # ----------------------------------------------------

        epochs = int(
            config.get(
                "epochs",
                3,
            )
        )

        learning_rate = float(
            config.get(
                "learning_rate",
                0.001,
            )
        )

        # ----------------------------------------------------
        # Differential Privacy configuration
        # ----------------------------------------------------

        dp_enabled_value = config.get(
            "differential_privacy",
            True,
        )

        # Handle Flower/string configuration safely
        if isinstance(
            dp_enabled_value,
            str,
        ):

            dp_enabled = (
                dp_enabled_value.lower()
                in [
                    "true",
                    "1",
                    "yes",
                ]
            )

        else:

            dp_enabled = bool(
                dp_enabled_value
            )

        target_epsilon = float(
            config.get(
                "target_epsilon",
                2.0,
            )
        )

        target_delta = float(
            config.get(
                "target_delta",
                1e-5,
            )
        )

        max_grad_norm = float(
            config.get(
                "max_grad_norm",
                1.0,
            )
        )

        # ----------------------------------------------------
        # Print configuration
        # ----------------------------------------------------

        print(
            "\n========== TRAINING CONFIGURATION =========="
        )

        print(
            f"Client: {self.cid}"
        )

        print(
            f"Epochs: {epochs}"
        )

        print(
            f"Learning rate: {learning_rate}"
        )

        print(
            f"Differential Privacy: "
            f"{dp_enabled}"
        )

        if dp_enabled:

            print(
                f"Target epsilon: "
                f"{target_epsilon}"
            )

            print(
                f"Target delta: "
                f"{target_delta}"
            )

            print(
                f"Max gradient norm: "
                f"{max_grad_norm}"
            )

        print(
            "============================================"
        )
        print(
            f"[DIAG][client {self.cid}] entering local training",
            flush=True,
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

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

            dp_enabled=dp_enabled,

            target_epsilon=target_epsilon,

            target_delta=target_delta,

            max_grad_norm=max_grad_norm,
        )
        print(
            f"[DIAG][client {self.cid}] local training complete",
            flush=True,
        )

        # ----------------------------------------------------
        # Training finished
        # ----------------------------------------------------

        print(
            f"\nClient {self.cid}: "
            f"Training Finished.",
            flush=True,
        )

        print(
            f"Client {self.cid}: "
            f"Final local accuracy: "
            f"{accuracies[-1]:.2f}%"
        )

        # ----------------------------------------------------
        # Privacy information
        # ----------------------------------------------------

        if dp_enabled:

            print(
                f"Client {self.cid}: "
                f"ε = "
                f"{privacy_metrics['epsilon']:.4f}"
            )

        # ----------------------------------------------------
        # Metrics returned to server
        # ----------------------------------------------------

        metrics = {
            "accuracy": float(
                accuracies[-1]
            ),
        }

        if dp_enabled:

            metrics.update(
                {
                    "epsilon":
                        float(
                            privacy_metrics[
                                "epsilon"
                            ]
                        ),

                    "delta":
                        float(
                            privacy_metrics[
                                "delta"
                            ]
                        ),

                    "noise_multiplier":
                        float(
                            privacy_metrics[
                                "noise_multiplier"
                            ]
                        ),

                    "max_grad_norm":
                        float(
                            privacy_metrics[
                                "max_grad_norm"
                            ]
                        ),
                }
            )

        # ----------------------------------------------------
        # Return parameters
        # ----------------------------------------------------

        return (

            self.get_parameters(
                config={}
            ),

            len(
                self.train_loader.dataset
            ),

            metrics,
        )

    # ========================================================
    # Evaluation
    # ========================================================

    def evaluate(
        self,
        parameters,
        config,
    ):

        print(
            f"\nClient {self.cid}: "
            f"Evaluating..."
        )

        self.set_parameters(
            parameters
        )

        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------
        #
        # This is currently LOCAL evaluation.
        #
        # It evaluates on the client's training
        # dataset and is NOT our final paper metric.
        #
        # We will replace this with global held-out
        # test-set evaluation after the baseline works.
        # ----------------------------------------------------

        metrics = evaluate_model(
            model=self.model,

            dataloader=self.train_loader,

            device=self.device,
        )

        print(
            f"Client {self.cid} | "
            f"Local Accuracy: "
            f"{metrics['accuracy']:.2f}% | "
            f"Local F1: "
            f"{metrics['f1']:.2f}%"
        )

        return (

            metrics["loss"],

            len(
                self.train_loader.dataset
            ),

            {
                "accuracy":
                    metrics["accuracy"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

                "f1":
                    metrics["f1"],
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

    client = FlowerClient(
        str(args.client_id)
    )

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()