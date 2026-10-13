from __future__ import annotations

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | None = None,
):
    """
    Evaluate a trained PyTorch model.

    Parameters
    ----------
    model:
        PyTorch model to evaluate.

    dataloader:
        DataLoader containing the evaluation dataset.

    device:
        CPU or CUDA device.

    Returns
    -------
    dict
        Contains:
        - loss
        - accuracy
        - precision
        - recall
        - f1
        - confusion_matrix
    """

    # ========================================================
    # Device
    # ========================================================

    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    # ========================================================
    # Evaluation
    # ========================================================

    with torch.no_grad():

        for batch_X, batch_y in dataloader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)

            # Loss
            loss = criterion(
                outputs,
                batch_y,
            )

            batch_size = batch_y.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            # Predictions
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
    # Empty Dataset Check
    # ========================================================

    if total_samples == 0:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    # ========================================================
    # Average Loss
    # ========================================================

    average_loss = (
        total_loss / total_samples
    )

    # ========================================================
    # Metrics
    # ========================================================

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
    # Return
    # ========================================================

    return {
        "loss": float(average_loss),

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "confusion_matrix": cm.tolist(),
    }