# Graph Condensation via Gradient Matching (GCond)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wajason/GCond-PyTorch-Implementation/blob/main/gcond.ipynb)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

This repository contains a reproduction and analysis of the **GCond** algorithm, originally proposed in the paper *"Graph Condensation for Graph Neural Networks"* (ICLR 2022).

Graph Condensation aims to synthesize a small-scale, synthetic graph $\mathcal{S}$ from a large-scale real graph $\mathcal{T}$, such that a GNN trained on $\mathcal{S}$ achieves comparable performance to one trained on $\mathcal{T}$. This implementation focuses on the **Gradient Matching** strategy using the **Cora** dataset.

## 🛠️ Key Implementation Details

* **Algorithm:** GCond (Gradient Matching)
* **Dataset:** Cora Citation Network
* **Framework:** PyTorch & PyTorch Geometric (PyG)
* **Hardware:** NVIDIA Tesla T4 (Google Colab)

## 📊 Experimental Results

### 1. Data Reduction
We applied a reduction rate ($r$) of **0.5 (50%)** to the training set.

* **Original Training Nodes:** 140
* **Condensed Synthetic Nodes:** 70
* **Observation:** The synthetic graph $\mathcal{S}$ successfully compresses the topological and feature information of the original 140 nodes into a compact $70 \times 70$ adjacency structure while maintaining training utility.

### 2. Performance & Analysis
The model achieved ~80% test accuracy using the condensed graph. However, we analyzed the **Model Coupling (Bi-level Optimization) Issue**:

* **Phenomenon:** When training a GNN architecture different from the one used during the condensation phase (e.g., changing layers, activation, or aggregation), the accuracy drops significantly (to ~30%).
* **Insight:** GCond relies on matching gradients for a *specific* set of parameters $\theta$. The synthetic graph $\mathcal{S}$ essentially encodes information tailored to "trick" a specific architecture. It acts as an "adversarial" shortcut for that specific model, leading to poor generalization across heterogeneous architectures.

## 📝 Reference

This work is based on the following paper:
> **Graph Condensation for Graph Neural Networks**
> Wei Jin, Lingxiao Zhao, Shichang Zhang, Yozen Liu, Jiliang Tang, Neil Shah
> *ICLR 2022*

Original Implementation: [https://github.com/ChandlerBang/GCond](https://github.com/ChandlerBang/GCond)
