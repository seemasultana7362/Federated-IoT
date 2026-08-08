# Privacy-Preserving Federated Learning for Secure IoT-Based Distributed Data Analytics

A research-driven implementation of a **Privacy-Preserving Federated Learning (FL)** framework for IoT intrusion detection that combines **Flower**, **PyTorch**, **Differential Privacy**, **Secure Aggregation**, and **Adaptive Communication**.

---

## Overview

Traditional machine learning requires collecting all client data on a centralized server, introducing major privacy and security concerns.

This project implements a decentralized Federated Learning framework where IoT devices collaboratively train an intrusion detection model without sharing raw data.

The framework integrates:

- Federated Learning (Flower)
- PyTorch-based Neural Network
- Differential Privacy
- Secure Aggregation
- Adaptive Communication
- IID and Non-IID Client Simulation

---

## Key Features

- Privacy-preserving collaborative learning
- Flower-based Federated Learning
- Differential Privacy using Opacus
- Secure Aggregation
- Adaptive client communication
- IoT Intrusion Detection
- IID & Non-IID experiments
- Modular architecture
- Fully reproducible research pipeline

---

## Project Architecture

```
                   Raw Dataset
                        │
                Data Preprocessing
                        │
               Feature Engineering
                        │
          IID / Non-IID Client Split
                        │
          ┌─────────────┴─────────────┐
          │                           │
     Flower Client 1            Flower Client N
          │                           │
      Local Training             Local Training
          │                           │
 Differential Privacy        Differential Privacy
          │                           │
 Secure Aggregation      Secure Aggregation
          └─────────────┬─────────────┘
                        │
                  Flower Server
                        │
                     FedAvg
                        │
              Updated Global Model
                        │
          Adaptive Communication
                        │
                  Performance Evaluation
```

---

## Repository Structure

```
Federated-IoT/
│
├── aggregation/
├── client/
├── communication/
├── config/
├── dataset/
├── evaluation/
├── models/
├── preprocessing/
├── privacy/
├── server/
├── results/
├── README.md
├── ProductAnalysis.md
├── TECH.md
├── requirements.txt
└── .gitignore
```

---

## Technology Stack

| Category | Technology |
|-----------|------------|
| Programming | Python 3.11 |
| Deep Learning | PyTorch |
| Federated Learning | Flower |
| Privacy | Opacus |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| IDE | VS Code |
| Version Control | Git & GitHub |

---

## Dataset

Primary Dataset

- UNSW-NB15

Future Dataset

- TON-IoT

---

## Experiments

- Centralized Learning
- Federated Learning (IID)
- Federated Learning (Non-IID)
- Differential Privacy
- Secure Aggregation
- Adaptive Communication

---

## Evaluation Metrics

Machine Learning

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

System Metrics

- Communication Overhead
- Bandwidth Usage
- Aggregation Time
- Training Time
- Memory Usage
- CPU Usage

---

## Research Contributions

- Privacy-preserving Federated Learning
- IoT Intrusion Detection
- Differential Privacy Integration
- Secure Aggregation
- Adaptive Communication Strategy
- Evaluation under IID & Non-IID environments

---

## Future Work

- Homomorphic Encryption
- Blockchain Integration
- Edge Computing
- Real IoT Deployment
- Transformer Models
- Asynchronous Federated Learning

---

## License

MIT License

---

## Author

Seema Sultana

Computer Science Engineering

Research Area:
Federated Learning • Privacy-Preserving AI • Distributed Machine Learning • Cybersecurity
