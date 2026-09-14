from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mlp import MLP

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


# ============================================================
# Build Model
# ============================================================

def build_client_model(
    input_size: int,
    device: torch.device | None = None,
) -> nn.Module:

    model = MLP(
        input_size=input_size
    )

    if device is not None:
        model.to(device)

    return model


# ============================================================
# Train Model
# ============================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 3,
    learning_rate: float = 0.001,
    device: torch.device | None = None,

    # Differential Privacy
    dp_enabled: bool = True,
    target_epsilon: float = 2.0,
    target_delta: float = 1e-5,
    max_grad_norm: float = 1.0,
):

    # ========================================================
    # Device
    # ========================================================

    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.train()

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # ========================================================
    # Privacy Variables
    # ========================================================

    privacy_engine = None
    privacy_metrics = {
        "epsilon": 0.0,
        "delta": 0.0,
        "max_grad_norm": 0.0,
        "noise_multiplier": 0.0,
    }

    # ========================================================
    # DIFFERENTIAL PRIVACY
    # ========================================================

    if dp_enabled:

        print(
            "\n[DP] Differential Privacy enabled."
        )

        # ----------------------------------------------------
        # Validate model
        # ----------------------------------------------------

        errors = ModuleValidator.validate(
            model,
            strict=False,
        )

        if errors:

            print(
                "\n[DP] Model compatibility issues:"
            )

            for error in errors:
                print(
                    f"    {error}"
                )

            raise ValueError(
                "Model is not compatible with Opacus."
            )

        print(
            "[DP] Model validation passed."
        )

        # ----------------------------------------------------
        # Create Privacy Engine
        # ----------------------------------------------------

        privacy_engine = PrivacyEngine(
            accountant="prv",
        )

        # ----------------------------------------------------
        # Convert training to DP-SGD
        # ----------------------------------------------------

        model, optimizer, train_loader = (
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=epochs,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                max_grad_norm=max_grad_norm,
            )
        )

        # ----------------------------------------------------
        # Get actual noise multiplier
        # ----------------------------------------------------

        noise_multiplier = (
            optimizer.noise_multiplier
        )

        print(
            f"[DP] Target epsilon: "
            f"{target_epsilon}"
        )

        print(
            f"[DP] Target delta: "
            f"{target_delta}"
        )

        print(
            f"[DP] Max gradient norm: "
            f"{max_grad_norm}"
        )

        print(
            f"[DP] Noise multiplier: "
            f"{noise_multiplier:.6f}"
        )

    # ========================================================
    # NON-DP BASELINE
    # ========================================================

    else:

        print(
            "\n[BASELINE] Differential Privacy disabled."
        )

        print(
            "[BASELINE] Standard PyTorch training."
        )

    # ========================================================
    # Training
    # ========================================================

    losses = []
    accuracies = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # ------------------------------------------------
            # Reset gradients
            # ------------------------------------------------

            optimizer.zero_grad()

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(batch_X)

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                outputs,
                batch_y,
            )

            # ------------------------------------------------
            # Backward pass
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Optimizer step
            # ------------------------------------------------

            optimizer.step()

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1,
            )

            total += batch_y.size(0)

            correct += (
                predicted == batch_y
            ).sum().item()

        # ====================================================
        # Epoch metrics
        # ====================================================

        epoch_loss = (
            running_loss / len(train_loader)
            if len(train_loader) > 0
            else 0.0
        )

        epoch_accuracy = (
            100.0 * correct / total
            if total > 0
            else 0.0
        )

        losses.append(
            epoch_loss
        )

        accuracies.append(
            epoch_accuracy
        )

        prefix = (
            "[DP]"
            if dp_enabled
            else "[BASELINE]"
        )

        print(
            f"{prefix} "
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_accuracy:.2f}%"
        )

    # ========================================================
    # Differential Privacy Accounting
    # ========================================================

    if dp_enabled:

        epsilon = (
            privacy_engine.get_epsilon(
                target_delta
            )
        )

        noise_multiplier = (
            optimizer.noise_multiplier
        )

        privacy_metrics = {
            "epsilon": epsilon,
            "delta": target_delta,
            "max_grad_norm": max_grad_norm,
            "noise_multiplier": noise_multiplier,
        }

        print(
            "\n[DP] Privacy accounting:"
        )

        print(
            f"    epsilon (ε): "
            f"{epsilon:.4f}"
        )

        print(
            f"    delta (δ): "
            f"{target_delta:.2e}"
        )

        print(
            f"    max_grad_norm: "
            f"{max_grad_norm}"
        )

        print(
            f"    noise_multiplier: "
            f"{noise_multiplier:.6f}"
        )

    # ========================================================
    # Return
    # ========================================================

    return (
        model,
        losses,
        accuracies,
        privacy_metrics,
    )