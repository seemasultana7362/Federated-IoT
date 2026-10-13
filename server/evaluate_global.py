from collections import OrderedDict
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from flwr.common import parameters_to_ndarrays

from models.mlp import MLP


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "test_processed.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "secure_aggregation"
)

ROUND_HISTORY_PATH = (
    RESULTS_DIR
    / "round_history.csv"
)

DEVICE = torch.device("cpu")

INPUT_SIZE = 42
BATCH_SIZE = 1024


def save_round_result(
    round_number,
    loss,
    accuracy,
    precision,
    recall,
    f1,
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = ROUND_HISTORY_PATH.exists()

    with open(
        ROUND_HISTORY_PATH,
        "a",
        newline="",
    ) as file:

        writer = csv.writer(file)

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


def load_global_test_data():

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Global test dataset not found:\n{TEST_PATH}"
        )

    print(
        f"[DATA] Loading global test dataset: {TEST_PATH}"
    )

    df = pd.read_csv(TEST_PATH)

    if "label" not in df.columns:
        raise ValueError(
            "Column 'label' not found in test dataset."
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in ["label", "attack_cat"]
    ]

    if len(feature_columns) != INPUT_SIZE:
        raise ValueError(
            f"Expected {INPUT_SIZE} features, "
            f"found {len(feature_columns)}."
        )

    X = df[feature_columns].values.astype(
        np.float32
    )

    y = df["label"].values.astype(
        np.int64
    )

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(
        f"[DATA] Global test set: "
        f"{len(dataset)} samples, "
        f"{len(feature_columns)} features"
    )

    return loader


def evaluate_global_model(
    server_round,
    parameters,
    config=None,
):

    print()
    print(
        f"[SERVER] Global evaluation - Round {server_round}"
    )

    # Flower FedAvg supplies Parameters here.
    if hasattr(parameters, "tensors"):
        ndarrays = parameters_to_ndarrays(parameters)
    else:
        ndarrays = parameters

    model = MLP(
        input_size=INPUT_SIZE
    )

    state_dict = model.state_dict()

    if len(ndarrays) != len(state_dict):
        raise ValueError(
            "Number of received parameter arrays does not "
            "match model state_dict."
        )

    new_state_dict = OrderedDict()

    for key, value in zip(
        state_dict.keys(),
        ndarrays,
    ):
        new_state_dict[key] = torch.tensor(
            value,
            dtype=state_dict[key].dtype,
        )

    model.load_state_dict(
        new_state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    test_loader = load_global_test_data()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_X)

            loss = criterion(
                outputs,
                batch_y,
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            batch_size = batch_y.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                batch_y.cpu().numpy()
            )

    average_loss = (
        total_loss
        / max(total_samples, 1)
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    precision = precision_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(
        all_targets,
        all_predictions,
    )

    print()
    print(
        f"Test Loss: {average_loss:.4f}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall: {recall * 100:.2f}%"
    )

    print(
        f"F1: {f1 * 100:.2f}%"
    )

    print()
    print("Confusion Matrix:")
    print(cm)

    save_round_result(
        round_number=server_round,
        loss=average_loss,
        accuracy=accuracy * 100,
        precision=precision * 100,
        recall=recall * 100,
        f1=f1 * 100,
    )

    metrics = {
        "accuracy": float(
            accuracy * 100
        ),
        "precision": float(
            precision * 100
        ),
        "recall": float(
            recall * 100
        ),
        "f1": float(
            f1 * 100
        ),
    }

    return (
        float(average_loss),
        metrics,
    )