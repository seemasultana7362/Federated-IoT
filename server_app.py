
from logging import INFO
from typing import Dict

from flwr.common import Context, log, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow

from communication.adaptive import AdaptiveFedAvg
from models.mlp import MLP
from server.evaluate_global import evaluate_global_model
from workflow_with_log import SecAggPlusWorkflowWithLogs


DEFAULT_NUM_CLIENTS = 5
INPUT_SIZE = 42


def parse_bool(value, default=False) -> bool:
    """Convert common string and native boolean values to bool."""
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError(f"Invalid boolean value: {value!r}")

    return bool(value)


def get_initial_parameters():
    model = MLP(input_size=INPUT_SIZE)

    parameters = [
        value.detach().cpu().numpy()
        for value in model.state_dict().values()
    ]

    return ndarrays_to_parameters(parameters)


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = context.run_config

    num_rounds = int(config["num-server-rounds"])
    num_clients = int(
        config.get("num-clients", DEFAULT_NUM_CLIENTS)
    )
    local_epochs = int(config["local-epochs"])
    learning_rate = float(config["learning-rate"])

    differential_privacy = parse_bool(
        config.get("differential-privacy", True),
        default=True,
    )
    target_epsilon = float(config.get("target-epsilon", 2.0))
    target_delta = float(config.get("target-delta", 1e-5))
    max_grad_norm = float(config.get("max-grad-norm", 1.0))

    num_shares = int(config.get("num-shares", 5))
    reconstruction_threshold = int(
        config.get("reconstruction-threshold", 3)
    )
    max_weight = float(config.get("max-weight", 1000.0))
    timeout = float(config.get("timeout", 120.0))

    secure_aggregation = parse_bool(
        config.get("secure-aggregation", False),
        default=False,
    )
    adaptive_communication = parse_bool(
        config.get("adaptive-communication", False),
        default=False,
    )

    if num_clients < 1:
        raise ValueError("num-clients must be at least 1")

    if adaptive_communication and secure_aggregation:
        raise ValueError(
            "Adaptive client participation must first be tested "
            "with secure-aggregation=false."
        )

    if secure_aggregation:
        if not 1 <= reconstruction_threshold <= num_shares:
            raise ValueError(
                "reconstruction-threshold must be between 1 "
                "and num-shares."
            )

    print("\n" + "=" * 70)
    print("FEDERATED LEARNING")
    print("=" * 70)
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Local epochs: {local_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Differential Privacy: {differential_privacy}")
    print(f"Target epsilon: {target_epsilon}")
    print(f"Target delta: {target_delta}")
    print(f"Max grad norm: {max_grad_norm}")
    print(f"Adaptive communication: {adaptive_communication}")
    print(f"Secure aggregation: {secure_aggregation}")
    if secure_aggregation:
        print(f"SecAgg+ shares: {num_shares}")
        print(
            "SecAgg+ reconstruction threshold: "
            f"{reconstruction_threshold}"
        )
    print("=" * 70 + "\n")

    log(INFO, "[DIAG][server] creating initial model parameters")
    initial_parameters = get_initial_parameters()
    log(INFO, "[DIAG][server] initial model parameters created")

    def fit_config(server_round: int) -> Dict:
        return {
            "server_round": server_round,
            "epochs": local_epochs,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "lr": learning_rate,
            "differential_privacy": differential_privacy,
            "privacy_enabled": differential_privacy,
            "dp_enabled": differential_privacy,
            "target_epsilon": target_epsilon,
            "target_delta": target_delta,
            "max_grad_norm": max_grad_norm,
        }

    strategy_kwargs = {
        "fraction_fit": 1.0,
        "min_fit_clients": num_clients,
        "min_available_clients": num_clients,
        "fraction_evaluate": 0.0,
        "initial_parameters": initial_parameters,
        "on_fit_config_fn": fit_config,
        "evaluate_fn": evaluate_global_model,
    }

    if adaptive_communication:
        strategy = AdaptiveFedAvg(
            total_clients=num_clients,
            min_clients=min(3, num_clients),
            initial_fraction=1.0,
            min_fraction=0.6,
            fraction_step=0.2,
            min_accuracy_improvement=0.1,
            **strategy_kwargs,
        )
    else:
        strategy = FedAvg(**strategy_kwargs)

    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    if secure_aggregation:
        fit_workflow = SecAggPlusWorkflowWithLogs(
            num_shares=num_shares,
            reconstruction_threshold=reconstruction_threshold,
            max_weight=max_weight,
            timeout=timeout,
        )
        workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    else:
        workflow = DefaultWorkflow()

    log(
        INFO,
        "[DIAG][server] starting federated workflow "
        "(secure_aggregation=%s, adaptive_communication=%s)",
        secure_aggregation,
        adaptive_communication,
    )

    workflow(grid, legacy_context)

    log(INFO, "[DIAG][server] federated workflow complete")
