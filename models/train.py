from pathlib import Path
import json
import time
import os

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from mlp import MLP


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "dataset" / "processed" / "train_processed.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"

RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

train = pd.read_csv(TRAIN_PATH)

print("Dataset Shape :", train.shape)

# Features and Labels

X = train.drop(columns=["label", "attack_cat"])

y = train["label"]

print("Feature Shape :", X.shape)
print("Label Shape   :", y.shape)


# ============================================================
# CONVERT TO TENSORS
# ============================================================

X = torch.tensor(X.values, dtype=torch.float32)

y = torch.tensor(y.values, dtype=torch.long)

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True
)

print("Number of batches :", len(loader))


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device :", device)


# ============================================================
# MODEL
# ============================================================

model = MLP(input_size=X.shape[1]).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print(model)


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 10

train_losses = []
train_accuracies = []

print("\nStarting Training...\n")

start_time = time.time()

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        leave=False
    )

    for X_batch, y_batch in progress:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += y_batch.size(0)

        correct += (predicted == y_batch).sum().item()

    epoch_loss = running_loss / len(loader)

    epoch_accuracy = 100 * correct / total

    train_losses.append(float(epoch_loss))

    train_accuracies.append(float(epoch_accuracy))

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS}"
        f" | Loss: {epoch_loss:.4f}"
        f" | Accuracy: {epoch_accuracy:.2f}%"
    )

training_time = time.time() - start_time


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = MODEL_DIR / "mlp_baseline.pth"

torch.save(model.state_dict(), MODEL_PATH)

print("\nModel Saved")

print(MODEL_PATH)


# ============================================================
# SAVE HISTORY
# ============================================================

history = {

    "epochs": EPOCHS,

    "loss": train_losses,

    "accuracy": train_accuracies,

    "training_time_seconds": float(training_time)

}

JSON_PATH = RESULTS_DIR / "baseline_history.json"

with open(JSON_PATH, "w") as f:

    json.dump(history, f, indent=4)

print("History Saved")

print(JSON_PATH)


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(train_losses, linewidth=2)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.savefig(RESULTS_DIR / "loss_curve.png")

plt.close()


# ============================================================
# ACCURACY CURVE
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(train_accuracies, linewidth=2)

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.grid(True)

plt.savefig(RESULTS_DIR / "accuracy_curve.png")

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("TRAINING COMPLETED")

print("=" * 60)

print(f"Epochs           : {EPOCHS}")

print(f"Final Accuracy   : {train_accuracies[-1]:.2f}%")

print(f"Final Loss       : {train_losses[-1]:.4f}")

print(f"Training Time    : {training_time:.2f} seconds")

print("=" * 60)