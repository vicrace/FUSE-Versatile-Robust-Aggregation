# Towards Practical FHE-Federated LLMs Training: FHE-Compatible Compression Techniques and Versatile Robust Aggregation

This repository contains the official implementation of **FUSE**, a unified FHE-native compression suite for FHE-based Federated Learning on Large Language Models. **FUSE-SPA** (Sparse Update Packing),  **FUSE-RDX** (Radix Dimension Reduction) and **FUSE-Fusion** (Hybrid Mode).

_and_

**Versatile Robust Aggregation**, a unified FHE-compatible aggregation algorithms supporting FedAvg, FedProx, and Norm-Bounding.

---
## 🛠️ Installation

### Prerequisites
- **Hardware:** NVIDIA GPU (RTX 3090/4090/5090) with CUDA 12.x.
- **Environment:** Linux (Ubuntu 22.04 recommended).
- **Dependencies:** Python 3.10+, OpenFHE v1.3.1, OpenFHE-python, OpenMP (libgomp1)

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
  bash scripts/training_scripts/run_plaintext.sh   # Non-encrypted FL training
  bash scripts/training_scripts/run_fhe_baseline.sh # FHE-baseline

- FUSE Variants:
  ```bash
  bash scripts/training_scripts/run_fuse_spa.sh    # Sparse Update Packing (Selective updates)
  bash scripts/training_scripts/run_fuse_rdx.sh    # Dimension Reduction (Compressed updates)
  bash scripts/training_scripts/run_fuse_fusion.sh # Hybrid Compression (Select then Compress)

## ⚙️ Shell Script Configurations

The `.sh` scripts include several configurable parameters:

- topk        — the top-k ratio used for sparse updates in FUSE.
- spa_bin     — the binning configuration for fuse_spa (Sparse Update Packing).
- rdx_scale   — the scaling factor for fuse_rdx (Dimension Reduction).
- fed_alg     — the federated learning algorithm to use. Options include:
    - fedavg        — standard FedAvg
    - fedprox       — FedProx algorithm
    - fednormbound  — Norm Bounding algorithm

---

## 🧪 Evaluation

For evaluation requires two main steps:

### 1. Merge LoRA Checkpoints

Use the script in utils/merge_lora.py to merge LoRA weights with the base model:

python utils/merge_lora.py --base_model_path [trained model] --lora_path [checkpoint files]

- --base_model_path  — path to the base model (e.g.,openai-community/gpt2) .
- --lora_path        — path to the LoRA checkpoint(s) generated during training.

### 2. Evaluate on Benchmark Datasets

- utils/evaluate_lambada.py — evaluates performance on the LAMBADA dataset.
- utils/evaluate_wiki103.py — evaluates performance on WikiText-103.
- utils/evaluate_1bw.py     — evaluates performance on the 1 Billion Word dataset.
