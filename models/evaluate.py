from pathlib import Path

import json

import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from mlp import MLP

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = PROJECT_ROOT / "dataset" / "processed" / "test_processed.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "saved_models" / "mlp_baseline.pth"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

test = pd.read_csv(TEST_PATH)

print(test.shape)

X_test = test.drop(columns=["label", "attack_cat"])

y_test = test["label"]

X_test = torch.tensor(
    X_test.values,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test.values,
    dtype=torch.long
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP(input_size=X_test.shape[1]).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

with torch.no_grad():

    outputs = model(X_test.to(device))

    predictions = torch.argmax(outputs, dim=1)

predictions = predictions.cpu()

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

print(classification_report(
    y_test,
    predictions
))

cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png"
)

plt.close()

metrics = {

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1)

}

with open(
    RESULTS_DIR / "baseline_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print("="*60)

print("EVALUATION COMPLETE")

print("="*60)

print(metrics)

print("="*60)