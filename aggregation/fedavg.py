from __future__ import annotations

from typing import Any

import flwr as fl


class FedAvgStrategy(fl.server.strategy.FedAvg):
    """Concrete FedAvg strategy wrapper used by the project.

    The project exercises a custom strategy class in several launch paths, so
    keeping this as a real subclass is preferable to leaving an empty stub.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("fraction_fit", 1.0)
        kwargs.setdefault("fraction_evaluate", 1.0)
        kwargs.setdefault("min_fit_clients", 1)
        kwargs.setdefault("min_evaluate_clients", 1)
        kwargs.setdefault("min_available_clients", 1)
        super().__init__(*args, **kwargs)
