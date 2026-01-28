# Towards Practical FHE-Federated LLMs Training: FHE-Compatible Compression Techniques and Versatile Robust Aggregation

This repository contains the official implementation of **FUSE**, a unified FHE-native compression suite for FHE-based Federated Learning on Large Language Models. **FUSE-SPA** (Sparse Update Packing),  **FUSE-RDX** (Radix Dimension Reduction) and **FUSE-Fusion** (Hybrid Mode).

_and_

**Versatile Robust Aggregation**, a unified FHE-compatible aggregation algorithms supporting FedAvg, FedProx, and Norm-Bounding.

---
## 🛠️ Installation

### Prerequisites
- **Hardware:** NVIDIA GPU (RTX 3090/4090/5090) with CUDA 12.x.
- **Environment:** Linux (Ubuntu 22.04 recommended).
- **Dependencies:** Python 3.10+, OpenFHE v1.2, OpenFHE-python, OpenMP (libgomp1)

### Setup
- Create a virtual environment and install requirements:
  ```bash
  conda create -n fuse python=3.10
  conda activate fuse
  pip install -r requirements.txt

---
## 🚀 Running Experiments
All experiment configurations are pre-configured in the provided shell scripts. 

- Standard Baseline (Plaintext/FHE):
  ```bash
  bash scripts/run_plaintext.sh   # Non-encrypted FL training
  bash scripts/run_fhe_baseline.sh # FHE-baseline

- FUSE Variants:
  ```bash
  bash scripts/run_fuse_spa.sh    # Sparse Update Packing (Selective updates)
  bash scripts/run_fuse_rdx.sh    # Dimension Reduction (Compressed updates)
  bash scripts/run_fuse_fusion.sh # Hybrid Compression (Select then Compress)
