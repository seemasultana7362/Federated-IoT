from __future__ import annotations

from typing import Any

import flwr as fl


class SecureAggregationStrategy(fl.server.strategy.FedAvg):
    """Compatibility shim for secure aggregation experiments.

    The project references this module as a secure aggregation strategy, so the
    minimal functional implementation keeps the class importable and configurable.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("fraction_fit", 1.0)
        kwargs.setdefault("fraction_evaluate", 1.0)
        kwargs.setdefault("min_fit_clients", 1)
        kwargs.setdefault("min_evaluate_clients", 1)
        kwargs.setdefault("min_available_clients", 1)
        super().__init__(*args, **kwargs)
