from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_client_dataset(client_id: int, data_dir: Path | None = None):
    if data_dir is None:
        data_dir = PROJECT_ROOT / "dataset" / "clients" / "iid"

    client_candidates = [
        data_dir / f"client_{client_id}.csv",
        data_dir / f"client_{client_id + 1}.csv",
        data_dir / f"client_{client_id + 2}.csv",
    ]

    client_path = next((path for path in client_candidates if path.exists()), None)
    if client_path is None:
        matching_files = sorted(p.name for p in data_dir.glob("client_*.csv"))
        raise FileNotFoundError(
            f"Client dataset not found for client_id={client_id} in {data_dir}. Available files: {matching_files}"
        )

    df = pd.read_csv(client_path)
    X = df.drop(columns=["label", "attack_cat"])
    y = df["label"]

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return loader, X.shape[1]
