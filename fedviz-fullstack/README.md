# FedViz Lab — Full-stack demo

Standalone Flask REST API + interactive browser UI, separate from the original Federated-IoT repository.

## Run on macOS
```bash
cd fedviz-fullstack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5173

## API
- `GET /api/health`
- `POST /api/experiment` with JSON such as:
```json
{"clients":12,"rounds":30,"participation":80,"noise":0.6,"clip":1.0,"threshold":0.2,"dataset":"UNSW-NB15","distribution":"Non-IID","seed":42}
```

## Link from your existing project
Deploy this app to a Python-capable host, then add the real public URL to the original repository README:
```md
## Interactive demo
[Launch FedViz Lab](https://YOUR-DEPLOYED-URL)
```
Do not use localhost as a public link. You can also keep the code in a separate GitHub repository and link it from your research repo.

## Research integrity
This is a parameter-reactive synthetic simulator. It does not train Flower/PyTorch models, load UNSW-NB15, run cryptographic secure aggregation, or calculate formal DP guarantees. Replace `run_experiment()` with your actual training pipeline before reporting results as research measurements.
