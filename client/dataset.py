from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Configuration
# ============================================================

# The dashboard sets FEDERATED_IOT_SPLIT to select a matching local partition.
# Default remains non-IID so existing CLI workflows keep their previous behavior.
_CLIENT_SPLIT = os.environ.get("FEDERATED_IOT_SPLIT", "non_iid").strip().lower()
if _CLIENT_SPLIT not in {"iid", "non_iid"}:
    raise ValueError("FEDERATED_IOT_SPLIT must be 'iid' or 'non_iid'.")

DEFAULT_CLIENT_DATA_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "clients"
    / _CLIENT_SPLIT
)

DEFAULT_TEST_DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "test_processed.csv"
)


# ============================================================
# Load Client Dataset
# ============================================================

def load_client_dataset(
    client_id: int,
    data_dir: Path | None = None,
):
    """
    Load one federated client's local training dataset.

    Expected format:

        dataset/clients/non_iid/
            client_1.csv
            client_2.csv
            client_3.csv
            client_4.csv
            client_5.csv

    Each CSV must contain:
        - label
        - optionally attack_cat

    Returns
    -------
    loader : DataLoader
        Local training DataLoader.

    input_size : int
        Number of input features.
    """

    if data_dir is None:
        data_dir = DEFAULT_CLIENT_DATA_DIR

    data_dir = Path(data_dir)

    client_path = (
        data_dir / f"client_{client_id}.csv"
    )

    if not client_path.exists():

        matching_files = sorted(
            p.name
            for p in data_dir.glob("client_*.csv")
        )

        raise FileNotFoundError(
            f"\nClient dataset not found.\n"
            f"Requested client: {client_id}\n"
            f"Expected file: {client_path}\n"
            f"Available files: {matching_files}"
        )

    print(
        f"[DATA] Loading client dataset: "
        f"{client_path}"
    )

    df = pd.read_csv(client_path)

    # --------------------------------------------------------
    # Validate label column
    # --------------------------------------------------------

    if "label" not in df.columns:

        raise ValueError(
            f"'label' column not found in "
            f"{client_path}"
        )

    # --------------------------------------------------------
    # Remove target / non-feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "label",
            "attack_cat",
        ]
    ]

    if len(feature_columns) == 0:

        raise ValueError(
            f"No feature columns found in "
            f"{client_path}"
        )

    X = df[feature_columns]
    y = df["label"]

    # --------------------------------------------------------
    # Validate numerical features
    # --------------------------------------------------------

    non_numeric_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if non_numeric_columns:

        raise ValueError(
            "Client dataset contains non-numeric "
            f"features: {non_numeric_columns}\n"
            "Your preprocessing pipeline must "
            "convert all features to numeric values "
            "before federated training."
        )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_tensor = torch.tensor(
        X.to_numpy(),
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y.to_numpy(),
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Create Dataset
    # --------------------------------------------------------

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    print(
        f"[DATA] Client {client_id}: "
        f"{len(dataset)} samples, "
        f"{X.shape[1]} features"
    )

    return loader, X.shape[1]


# ============================================================
# Load Global Test Dataset
# ============================================================

def load_test_dataset(
    test_path: Path | None = None,
):
    """
    Load the global held-out UNSW-NB15 test dataset.

    IMPORTANT:
    The test dataset must already have gone through
    the SAME preprocessing pipeline as the client datasets.

    Expected file:

        dataset/processed/test_processed.csv

    Required:
        label

    Optional:
        attack_cat

    Returns
    -------
    loader : DataLoader
        Global test DataLoader.

    input_size : int
        Number of input features.
    """

    if test_path is None:
        test_path = DEFAULT_TEST_DATA_PATH

    test_path = Path(test_path)

    if not test_path.exists():

        raise FileNotFoundError(
            "\nGlobal test dataset not found.\n\n"
            f"Expected:\n"
            f"    {test_path}\n\n"
            "Create test_processed.csv using the "
            "same preprocessing pipeline used for "
            "the client datasets before running "
            "final evaluation."
        )

    print(
        f"[DATA] Loading global test dataset: "
        f"{test_path}"
    )

    df = pd.read_csv(test_path)

    # --------------------------------------------------------
    # Validate label
    # --------------------------------------------------------

    if "label" not in df.columns:

        raise ValueError(
            f"'label' column not found in "
            f"{test_path}"
        )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "label",
            "attack_cat",
        ]
    ]

    if len(feature_columns) == 0:

        raise ValueError(
            f"No feature columns found in "
            f"{test_path}"
        )

    X = df[feature_columns]
    y = df["label"]

    # --------------------------------------------------------
    # Validate numerical features
    # --------------------------------------------------------

    non_numeric_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if non_numeric_columns:

        raise ValueError(
            "Test dataset contains non-numeric "
            f"features: {non_numeric_columns}\n"
            "The test set must use the same "
            "numerical feature representation as "
            "the client datasets."
        )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_tensor = torch.tensor(
        X.to_numpy(),
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y.to_numpy(),
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Create Dataset
    # --------------------------------------------------------

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
    )

    print(
        f"[DATA] Global test set: "
        f"{len(dataset)} samples, "
        f"{X.shape[1]} features"
    )

    return loader, X.shape[1]