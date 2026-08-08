import os
import json
import joblib
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs("../dataset/processed", exist_ok=True)
os.makedirs("./artifacts", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading datasets...")
print("=" * 60)

train = pd.read_csv("../dataset/raw/UNSW_NB15_training-set.csv")
test = pd.read_csv("../dataset/raw/UNSW_NB15_testing-set.csv")

print(f"Training Shape : {train.shape}")
print(f"Testing Shape  : {test.shape}")

# ============================================================
# REMOVE DUPLICATES
# ============================================================

print("\nRemoving duplicate rows...")

train = train.drop_duplicates()
test = test.drop_duplicates()

print(f"Training Shape : {train.shape}")
print(f"Testing Shape  : {test.shape}")

# ============================================================
# DROP ID COLUMN
# ============================================================

if "id" in train.columns:
    train.drop(columns=["id"], inplace=True)

if "id" in test.columns:
    test.drop(columns=["id"], inplace=True)

print("\nRemoved 'id' column.")

# ============================================================
# SAVE ATTACK CATEGORY
# (Used later for Non-IID client partitioning)
# ============================================================

train_attack = train["attack_cat"].copy()
test_attack = test["attack_cat"].copy()

# ============================================================
# DEFINE FEATURES
# ============================================================

TARGET = "label"

CATEGORICAL = [
    "proto",
    "service",
    "state"
]

# ============================================================
# ORDINAL ENCODER
# ============================================================

print("\nEncoding categorical features...")

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

train[CATEGORICAL] = encoder.fit_transform(train[CATEGORICAL])

test[CATEGORICAL] = encoder.transform(test[CATEGORICAL])

joblib.dump(
    encoder,
    "./artifacts/ordinal_encoder.pkl"
)

print("Categorical encoding completed.")

# ============================================================
# CREATE FEATURES AND LABELS
# ============================================================

X_train = train.drop(columns=[TARGET, "attack_cat"])
y_train = train[TARGET]

X_test = test.drop(columns=[TARGET, "attack_cat"])
y_test = test[TARGET]

feature_names = X_train.columns.tolist()

# ============================================================
# FEATURE SCALING
# ============================================================

print("\nScaling numerical features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    "./artifacts/scaler.pkl"
)

print("Scaling completed.")

# ============================================================
# CONVERT BACK TO DATAFRAME
# ============================================================

train_processed = pd.DataFrame(
    X_train_scaled,
    columns=feature_names
)

test_processed = pd.DataFrame(
    X_test_scaled,
    columns=feature_names
)

# ============================================================
# ADD LABEL BACK
# ============================================================

train_processed["label"] = y_train.values
test_processed["label"] = y_test.values

# ============================================================
# SAVE ATTACK CATEGORY
# ============================================================

train_processed["attack_cat"] = train_attack.values
test_processed["attack_cat"] = test_attack.values

# ============================================================
# SAVE DATASETS
# ============================================================

print("\nSaving processed datasets...")

train_processed.to_csv(
    "../dataset/processed/train_processed.csv",
    index=False
)

test_processed.to_csv(
    "../dataset/processed/test_processed.csv",
    index=False
)

# ============================================================
# SAVE ATTACK CATEGORY
# ============================================================

train_attack.to_csv(
    "../dataset/processed/train_attack.csv",
    index=False
)

test_attack.to_csv(
    "../dataset/processed/test_attack.csv",
    index=False
)

# ============================================================
# SAVE FEATURE NAMES
# ============================================================

joblib.dump(
    feature_names,
    "./artifacts/features.pkl"
)

# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "target_column": TARGET,
    "categorical_columns": CATEGORICAL,
    "removed_columns": [
        "id"
    ],
    "saved_attack_column": "attack_cat",
    "number_of_features": len(feature_names)
}

with open(
    "./artifacts/metadata.json",
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=4
    )

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(f"Processed Train Shape : {train_processed.shape}")
print(f"Processed Test Shape  : {test_processed.shape}")

print("\nSaved Files")

print("dataset/processed/train_processed.csv")
print("dataset/processed/test_processed.csv")
print("dataset/processed/train_attack.csv")
print("dataset/processed/test_attack.csv")

print("\nArtifacts")

print("ordinal_encoder.pkl")
print("scaler.pkl")
print("features.pkl")
print("metadata.json")

print("=" * 60)