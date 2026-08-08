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

    Returns
    -------
    dict
        {
            "loss": float,
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float,
            "confusion_matrix": ndarray
        }
    """

    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch_X, batch_y in dataloader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(batch_y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(all_labels, all_predictions) * 100

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    ) * 100

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    ) * 100

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    ) * 100

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }