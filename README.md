# VARG

Official implementation of the paper: **"Visual Autoregressive Modeling for Handwritten Mathematical Expression Generation"**.

---

## 🚀 Overview

**VARG** is a pioneering framework for Handwritten Mathematical Expression Generation (HMEG). It introduces a **visual autoregressive generation mechanism** to address the challenges of modeling complex spatial structures in mathematical formulas while preserving consistent handwriting styles.

By integrating hierarchical content encoding with style-aware transformation, VARG synthesizes high-quality HME images that serve as effective data augmentation for downstream recognition tasks.

---

## ⭐️ Key Contributions

* **Visual Autoregressive Mechanism**: The first HMEG method to introduce a visual autoregressive generation paradigm, enabling superior modeling of the two-dimensional spatial layout of expressions.
* **Style-Aware Transformer (SAT)**: A specialized module that progressively refines content representations under the explicit guidance of style conditions, ensuring stylistic coherence across the generated formula.
* **Hierarchical Content Encoding Module (HCEM)**:
    * **Hierarchical Extraction Unit (HEU)**: Captures the inherent structural dependencies and multi-scale features within mathematical expressions.
    * **Cross-Attention Mechanism (CAM)**: Employs a gated mechanism to dynamically suppress redundant features and enhance structural consistency.
* **SOTA Performance**: Outperforms existing state-of-the-art models across four evaluation metrics on benchmark datasets.
* **Downstream Boosting**: Proves that synthesized images significantly improve the performance of HMER systems via data augmentation.

---

## 🏗️ Architecture

The VARG framework consists of a **style encoder**, a **hierarchical content encoder**, and an **autoregressive transformer decoder**. The interaction between SAT and HCEM ensures that the generated output is both mathematically accurate and stylistically consistent.



---

## 📝 Datasets

The model is rigorously evaluated on the following benchmark datasets:
* **CROHME 2014 / 2016 / 2019**

---

## 🔥 Training

CUDA_VISIBLE_DEVICES=0,1,2,3 train.py


---

## 📂 Project Structure

```
.
├── configs/            # Experiment configurations (YAML)
├── data/               # Dataset loading and data augmentation
├── models/             # Model implementations (VARG, SAT, HCEM)
├── utils/              # Metrics, logging, and visualization utilities
└── train.py            # Main training entry

```


---

## 📌 Acknowledgments
We thank the open-source community for the foundational tools and datasets used in this research.

More details will be updated soon.
