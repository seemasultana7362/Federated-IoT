from __future__ import annotations

from collections import OrderedDict

import csv
import os
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from client.dataset import load_test_dataset
from models.mlp import MLP


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Results Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENT_NAME = os.environ.get(
    "FL_EXPERIMENT",
    "current",
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "dp"
    / EXPERIMENT_NAME
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ROUND_HISTORY_PATH = (
    RESULTS_DIR
    / "round_history.csv"
)


# ============================================================
# Save Round Result
# ============================================================

def save_round_result(
    round_number,
    loss,
    accuracy,
    precision,
    recall,
    f1,
):
    """
    Automatically save global test metrics
    after every federated round.
    """

    file_exists = ROUND_HISTORY_PATH.exists()

    with open(
        ROUND_HISTORY_PATH,
        "a",
        newline="",
    ) as file:

        writer = csv.writer(file)

        # ----------------------------------------------------
        # CSV Header
        # ----------------------------------------------------

        if not file_exists:

            writer.writerow(
                [
                    "round",
                    "loss",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                ]
            )

        # ----------------------------------------------------
        # Round Result
        # ----------------------------------------------------

        writer.writerow(
            [
                round_number,
                loss,
                accuracy,
                precision,
                recall,
                f1,
            ]
        )

    print(
        f"[RESULTS] Round {round_number} "
        f"saved to:"
    )

    print(
        f"          {ROUND_HISTORY_PATH}"
    )


# ============================================================
# Global Model Evaluation
# ============================================================

def evaluate_global_model(
    server_round,
    parameters,
    config = None,
):
    """
    Evaluate the federated global model on the
    held-out UNSW-NB15 test dataset.

    Flower calls this automatically after each
    federated round.
    """

    print(
        f"\n[SERVER] Global evaluation - "
        f"Round {server_round}"
    )

    # ========================================================
    # Load Test Dataset
    # ========================================================

    test_loader, input_size = (
        load_test_dataset()
    )

    print(
        f"[DATA] Global test set: "
        f"{len(test_loader.dataset)} samples, "
        f"{input_size} features"
    )

    # ========================================================
    # Validate Input Size
    # ========================================================

    if input_size != 42:

        raise ValueError(
            f"Expected 42 features, "
            f"but received {input_size}."
        )

    # ========================================================
    # Build Model
    # ========================================================

    model = MLP(
        input_size=input_size
    )

    model.to(DEVICE)

    # ========================================================
    # Convert Flower Parameters
    # ========================================================

    params_dict = zip(
        model.state_dict().keys(),
        parameters,
    )

    state_dict = OrderedDict(
        {
            key: torch.tensor(value)
            for key, value in params_dict
        }
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    # ========================================================
    # Evaluation Mode
    # ========================================================

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    # ========================================================
    # Inference
    # ========================================================

    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_X)

            loss = criterion(
                outputs,
                batch_y,
            )

            batch_size = batch_y.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            _, predicted = torch.max(
                outputs,
                1,
            )

            all_labels.extend(
                batch_y.cpu().numpy()
            )

            all_predictions.extend(
                predicted.cpu().numpy()
            )

    # ========================================================
    # Safety Check
    # ========================================================

    if total_samples == 0:

        raise ValueError(
            "Global test dataset is empty."
        )

    # ========================================================
    # Metrics
    # ========================================================

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        accuracy_score(
            all_labels,
            all_predictions,
        )
        * 100
    )

    precision = (
        precision_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
        * 100
    )

    recall = (
        recall_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
        * 100
    )

    f1 = (
        f1_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
        * 100
    )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    # ========================================================
    # Print Results
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        f"        GLOBAL TEST SET RESULTS - "
        f"ROUND {server_round}"
    )

    print(
        "=============================================="
    )

    print(
        f"Test Loss      : {average_loss:.4f}"
    )

    print(
        f"Accuracy       : {accuracy:.2f}%"
    )

    print(
        f"Precision      : {precision:.2f}%"
    )

    print(
        f"Recall         : {recall:.2f}%"
    )

    print(
        f"F1 Score       : {f1:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    print(
        "==============================================\n"
    )

    # ========================================================
    # Automatically Save Round
    # ========================================================

    save_round_result(
        round_number=server_round,
        loss=average_loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )

    # ========================================================
    # Return to Flower
    # ========================================================

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    return (
        float(average_loss),
        metrics,
    )