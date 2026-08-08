from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

client = pd.read_csv(
    PROJECT_ROOT / "dataset" / "clients" / "iid" / "client_1.csv"
)

print(client.shape)

print(client["label"].value_counts())