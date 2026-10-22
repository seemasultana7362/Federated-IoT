from logging import INFO
from typing import Dict

import torch

from flwr.common import Context, log, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow

from communication.adaptive import AdaptiveFedAvg
from models.mlp import MLP
from server.evaluate_global import evaluate_global_model
from workflow_with_log import SecAggPlusWorkflowWithLogs
#from workflow_with_log import SecAggPlusWorkflowWithLogs


NUM_CLIENTS = 5
INPUT_SIZE = 42


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

    num_rounds = int(context.run_config["num-server-rounds"])
    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])

    differential_privacy = bool(
        context.run_config["differential-privacy"]
    )

    target_epsilon = float(
        context.run_config["target-epsilon"]
    )

    target_delta = float(
        context.run_config["target-delta"]
    )

    max_grad_norm = float(
        context.run_config["max-grad-norm"]
    )

    num_shares = int(
        context.run_config["num-shares"]
    )

    reconstruction_threshold = int(
        context.run_config["reconstruction-threshold"]
    )

    max_weight = float(
        context.run_config["max-weight"]
    )

    timeout = float(
        context.run_config["timeout"]
    )


    num_clients = int(
        context.run_config.get("num-clients", NUM_CLIENTS)
    )

    secure_aggregation = context.run_config.get(
        "secure-aggregation", True
    )
    if isinstance(secure_aggregation, str):
        secure_aggregation = (
            secure_aggregation.strip().lower()
            in {"true", "1", "yes"}
        )
    else:
        secure_aggregation = bool(secure_aggregation)

    adaptive_communication = context.run_config.get(
        "adaptive-communication", False
    )
    if isinstance(adaptive_communication, str):
        adaptive_communication = (
            adaptive_communication.strip().lower()
            in {"true", "1", "yes"}
        )
    else:
        adaptive_communication = bool(adaptive_communication)

    if adaptive_communication and secure_aggregation:
        raise ValueError(
            "Adaptive client participation must first be tested "
            "with secure-aggregation=false."
        )


    print("\n" + "=" * 70)
    print("FEDERATED LEARNING - SECURE AGGREGATION")
    print("=" * 70)
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {num_rounds}")
    print(f"Local epochs: {local_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Differential Privacy: {differential_privacy}")
    print(f"Target epsilon: {target_epsilon}")
    print(f"Target delta: {target_delta}")
    print(f"Max grad norm: {max_grad_norm}")
    print(f"SecAgg+ shares: {num_shares}")
    print(
        f"SecAgg+ reconstruction threshold: "
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


    strategy_kwargs = dict(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_evaluate=0.0,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_global_model,
    )



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

        workflow = DefaultWorkflow(
            fit_workflow=fit_workflow,
        )
    else:
        workflow = DefaultWorkflow()



    log(INFO, "[DIAG][server] starting SecAgg+ workflow")
    workflow(grid, legacy_context)
    log(INFO, "[DIAG][server] federated workflow complete")
