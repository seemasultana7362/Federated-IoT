from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mlp import MLP


def build_client_model(input_size: int, device: torch.device | None = None) -> nn.Module:
    model = MLP(input_size=input_size)
    if device is not None:
        model.to(device)
    return model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 3,
    learning_rate: float = 0.001,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total if total > 0 else 0.0
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

    return model, losses, accuracies


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device | None = None):
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    if total == 0:
        return 0.0

    accuracy = 100.0 * correct / total
    return accuracy
