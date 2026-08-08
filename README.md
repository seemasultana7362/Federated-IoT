# Privacy-Preserving Federated Learning for IoT

## Dataset
TON-IoT

## Model
MLP

## Framework
Flower Federated Learning

## Enhancements
- Differential Privacy
- Secure Aggregation
- Adaptive Communication

## Final structure 
Federated-IoT

│

├── aggregation
│      fedavg.py
│      secure_aggregation.py
│
├── artifacts
│      models/
│      logs/
│      plots/
│
├── client
│      client.py
│      dataset.py
│      train.py
│      evaluate.py
│
├── communication
│      adaptive.py
│
├── config
│      config.yaml
│
├── dataset
│      clients/
│      processed/
│      raw/
│
├── evaluation
│      metrics.py
│
├── models
│      mlp.py
│
├── preprocessing
│
├── server
│      server.py
│      strategy.py
│
├── notebooks
│
├── main.py
│
├── requirements.txt
│
└── README.md