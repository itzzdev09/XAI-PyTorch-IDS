# Multi-Class Network Intrusion Detection Using CatBoost

![Project Banner](https://img.shields.io/badge/Status-Experimental-orange) ![Python](https://img.shields.io/badge/Python-3.10-blue) ![CatBoost](https://img.shields.io/badge/Framework-CatBoost-green)

---

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Training Details](#training-details)
* [Evaluation & Metrics](#evaluation--metrics)
* [SHAP Feature Importance](#shap-feature-importance)
* [Folder Structure](#folder-structure)
* [Contributing](#contributing)
* [License](#license)

---

## Project Overview

This project implements a **multi-class network intrusion detection system** using **CatBoost**. It is designed to handle large-scale datasets with extreme class imbalance.

The system includes:

* **Downsampling** for majority classes to balance the dataset.
* **K-Fold Cross-Validation** for robust evaluation.
* **GPU & CPU training support** with fallback.
* **SHAP explainability** to understand feature contributions.

---

## Dataset

* **Format**: Parquet (`.parquet`)
* **Source**: Processed network intrusion dataset (`simargl_full.parquet`)
* **Columns**: 33 features including `Label` (target) and categorical features like `PROTOCOL_MAP`, `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `ALERT`.
* **Classes**: Multi-class (e.g., 0, 1, 2, 3) with heavy imbalance.

---

## Features

* **Multi-class classification** using CatBoost
* **Downsampling** of majority classes to a defined max per class
* **K-Fold Cross-Validation** (`k=5`)
* **GPU acceleration with CPU fallback**
* **SHAP-based feature importance visualization**
* **Confusion matrix and metrics saving** for each fold

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/itzzdev09/catboost-network-detection.git
   cd catboost-network-detection
   ```
2. Create and activate a conda environment:

   ```bash
   conda create -n torchgpu python=3.10 -y
   conda activate torchgpu
   ```
3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

   **Requirements include:**

   * `catboost`
   * `numpy`
   * `pandas`
   * `scikit-learn`
   * `matplotlib`
   * `shap`

---

## Usage

To run the full training pipeline with K-Fold and SHAP:

```bash
python main_catboost.py
```

**Options you can configure in `main_catboost.py`:**

* `PER_CLASS_MAX` – Max rows per class
* `CB_PARAMS_GPU` – CatBoost GPU parameters
* `CB_PARAMS_CPU` – CatBoost CPU parameters
* `CHUNK_ITERS` – Chunked iterations for progress tracking
* `TEST_SIZE` – Validation split

---

## Training Details

* **Chunked training**: Trees are trained in blocks (default 200) for easier monitoring.
* **GPU first, CPU fallback**: Automatically falls back if GPU training fails.
* **Auto class weighting**: Uses `SqrtBalanced` to handle class imbalance.

---

## Evaluation & Metrics

* **Metrics saved**: `metrics.json` containing per-class precision, recall, F1-score, and overall accuracy.
* **Confusion matrices**: Saved per fold (`confusion_matrix_fold_X.png`)
* **Downsampling**: Maintains minority classes while limiting majority classes to `PER_CLASS_MAX`.

---

## SHAP Feature Importance

* SHAP values are computed using CatBoost’s **native SHAP** method.
* Summary plots are saved as `shap_summary.png`.
* Helps interpret the contribution of each feature toward the model’s predictions.

---

## Folder Structure

```
├── data/
│   └── processed/
│       └── simargl_full.parquet
├── main_catboost.py          # Training & evaluation pipeline
├── Runs/                     # Auto-created run directories
│   └── Run_<timestamp>/
│       ├── catboost_model.cbm
│       ├── metrics.json
│       ├── confusion_matrix_fold_X.png
│       ├── shap_summary.png
│       └── catboost_train_logs/
├── requirements.txt
└── README.md
```

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is **MIT Licensed** – see the `LICENSE` file for details.
