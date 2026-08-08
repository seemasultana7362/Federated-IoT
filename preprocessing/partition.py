from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "dataset" / "processed" / "train_processed.csv"
ATTACK_PATH = PROJECT_ROOT / "dataset" / "processed" / "train_attack.csv"

IID_PATH = PROJECT_ROOT / "dataset" / "clients" / "iid"
NONIID_PATH = PROJECT_ROOT / "dataset" / "clients" / "non_iid"

for client_dir in [IID_PATH, NONIID_PATH]:
    client_dir.mkdir(parents=True, exist_ok=True)
    for old_file in client_dir.glob("client_*.csv"):
        old_file.unlink()

print("Loading processed dataset...")

df = pd.read_csv(TRAIN_PATH)
attack = pd.read_csv(ATTACK_PATH)

if attack.shape[1] == 1:
    attack_series = attack.iloc[:, 0]
else:
    attack_series = attack["attack_cat"] if "attack_cat" in attack.columns else attack.iloc[:, 0]

if "attack_cat" not in df.columns:
    df["attack_cat"] = attack_series.values
else:
    df["attack_cat"] = attack_series.values

print("Processed dataset shape:", df.shape)

N_CLIENTS = 5

print("\nCreating IID clients...")
skf = StratifiedKFold(
    n_splits=N_CLIENTS,
    shuffle=True,
    random_state=42,
)

for client_id, (_, test_idx) in enumerate(
    skf.split(df, df["label"]),
    start=1,
):
    client_df = df.iloc[test_idx].copy().sample(frac=1, random_state=42 + client_id)
    client_df.to_csv(IID_PATH / f"client_{client_id}.csv", index=False)
    print(f"IID Client {client_id}: {client_df.shape}")

print("\nCreating non-IID clients by attack category...")

category_groups = []
for attack_name, attack_df in df.groupby("attack_cat", sort=False):
    category_groups.append((attack_name, attack_df))

category_groups.sort(key=lambda item: (-len(item[1]), item[0]))

client_frames = {client_id: [] for client_id in range(1, N_CLIENTS + 1)}

for group_idx, (attack_name, attack_df) in enumerate(category_groups):
    client_id = (group_idx % N_CLIENTS) + 1
    client_frames[client_id].append(attack_df)

for client_id in range(1, N_CLIENTS + 1):
    client_df = pd.concat(client_frames[client_id], axis=0).copy()
    client_df = client_df.sample(frac=1, random_state=100 + client_id)
    client_df.to_csv(NONIID_PATH / f"client_{client_id}.csv", index=False)
    print(
        f"Non-IID Client {client_id}: {client_df.shape} | "
        f"attack_cat={list(client_df['attack_cat'].value_counts().to_dict().items())}"
    )

print("\nVerification of non-IID clients:")
for client_id in range(1, N_CLIENTS + 1):
    client = pd.read_csv(NONIID_PATH / f"client_{client_id}.csv")
    print(f"Client {client_id}: shape={client.shape}, attack_cat_counts={client['attack_cat'].value_counts().to_dict()}")
