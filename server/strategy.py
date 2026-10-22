from __future__ import annotations

from typing import Any

import flwr as fl

from server.evaluate_global import evaluate_global_model


class FedAvgStrategy(fl.server.strategy.FedAvg):

    def __init__(self, *args: Any, **kwargs: Any):

        # ----------------------------------------------------
        # Automatically evaluate the global model after
        # every federated round.
        # ----------------------------------------------------

        kwargs.setdefault("evaluate_fn", evaluate_global_model)

        super().__init__(
            *args,
            **kwargs,
        )