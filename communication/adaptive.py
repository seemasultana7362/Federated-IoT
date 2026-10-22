
"""Adaptive client participation for federated IoT intrusion detection.

Adjusts the fraction of participating clients based on the improvement
in centralized global-test accuracy between consecutive rounds.

Accuracy returned by this project's evaluate_global_model is in
percentage points (0-100), not a fraction (0-1).
"""

from __future__ import annotations

from typing import Optional


from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class AdaptiveFedAvg(FedAvg):
    """FedAvg with accuracy-driven adaptive client participation."""

    def __init__(
        self,
        *,
        total_clients: int,
        min_clients: int = 3,
        initial_fraction: float = 1.0,
        min_fraction: float = 0.6,
        fraction_step: float = 0.2,
        min_accuracy_improvement: float = 0.1,
        **kwargs,
    ) -> None:
        if total_clients < 1:
            raise ValueError("total_clients must be at least 1")

        if not 1 <= min_clients <= total_clients:
            raise ValueError(
                "min_clients must be between 1 and total_clients"
            )

        if not 0 < min_fraction <= initial_fraction <= 1:
            raise ValueError(
                "Require 0 < min_fraction <= initial_fraction <= 1"
            )

        if not 0 < fraction_step <= 1:
            raise ValueError("fraction_step must be in (0, 1]")

        if min_accuracy_improvement < 0:
            raise ValueError(
                "min_accuracy_improvement cannot be negative"
            )

        self.total_clients = total_clients
        self.min_clients = min_clients
        self.min_fraction = min_fraction
        self.fraction_step = fraction_step
        self.min_accuracy_improvement = min_accuracy_improvement

        self.current_fraction = initial_fraction
        self.previous_accuracy: Optional[float] = None

        # Initialize the inherited FedAvg strategy.
        kwargs["fraction_fit"] = initial_fraction
        kwargs["min_fit_clients"] = min_clients
        kwargs["min_available_clients"] = total_clients

        super().__init__(**kwargs)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Select clients using the participation fraction for this round."""

        # FedAvg uses this fraction when calculating the sample size.
        self.fraction_fit = self.current_fraction

        instructions = super().configure_fit(
            server_round,
            parameters,
            client_manager,
        )

        available = client_manager.num_available()
        selected = len(instructions)

        print(
            f"[ADAPTIVE-COMM] Round {server_round}: "
            f"selected={selected}/{available}, "
            f"participation={self.current_fraction:.2f}"
        )

        return instructions

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[tuple[float, dict]]:
        """Evaluate globally and choose participation for the next round."""

        result = super().evaluate(server_round, parameters)

        # No centralized evaluation result is available.
        if result is None:
            print(
                "[ADAPTIVE-COMM] No global evaluation result; "
                "participation remains unchanged."
            )
            return None

        loss, metrics = result
        accuracy_value = metrics.get("accuracy")

        if accuracy_value is None:
            print(
                "[ADAPTIVE-COMM] Global accuracy missing; "
                "participation remains unchanged."
            )
            return result

        accuracy = float(accuracy_value)

        # Guard against invalid evaluation values.
        if not 0.0 <= accuracy <= 100.0:
            raise ValueError(
                "Expected global accuracy as a percentage in [0, 100]; "
                f"received {accuracy}."
            )

        if self.previous_accuracy is None:
            print(
                f"[ADAPTIVE-COMM] Round {server_round}: "
                f"initial global accuracy={accuracy:.4f}%."
            )
        else:
            improvement = accuracy - self.previous_accuracy

            if improvement >= self.min_accuracy_improvement:
                # Accuracy improved enough: reduce participation
                # for the next round, subject to the configured floor.
                self.current_fraction = max(
                    self.min_fraction,
                    self.current_fraction - self.fraction_step,
                )
                decision = "reduced"
            else:
                # Accuracy plateaued or dropped: increase participation.
                self.current_fraction = min(
                    1.0,
                    self.current_fraction + self.fraction_step,
                )
                decision = "increased"

            print(
                f"[ADAPTIVE-COMM] Round {server_round}: "
                f"accuracy={accuracy:.4f}%, "
                f"improvement={improvement:+.4f} percentage points; "
                f"participation {decision} to "
                f"{self.current_fraction:.2f} for the next round."
            )

        self.previous_accuracy = accuracy
        return result
