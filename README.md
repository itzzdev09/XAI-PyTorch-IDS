# An Explainable Ensemble-Based Framework for Binary Network Intrusion Detection

A stacking-ensemble Network Intrusion Detection System (NIDS) for **binary**
classification of network flows (BENIGN vs. ATTACK) on the **CICIDS2017**
benchmark, with explainability built into the detection pipeline via SHAP,
PDP and ICE.

## Framework

```
CICIDS2017 CSVs
      |
Cleaning & binary labeling  (BENIGN -> 0, everything else -> 1)
      |
Feature selection  (top-15 by variance)
      |
StandardScaler normalisation
      |
Train / test split  (80/20, stratified)
      |
+--------------- Level-0 base learners ---------------+
|  Random Forest  |  LightGBM  |  KNN  |  AdaBoost*   |
+-----------------------------------------------------+
      |
Level-1 meta-learner: Logistic Regression
      |
Final ensemble prediction  +  SHAP / PDP / ICE explanations
```

\* AdaBoost is currently trained and evaluated as a **standalone baseline
only** — it is not one of the stacked estimators. See *Known deviations* below.

## Results (CICIDS2017, binary)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Kappa | MCC |
|---|---|---|---|---|---|---|---|
| AdaBoost | 0.9018 | 0.9260 | 0.5443 | 0.6856 | 0.9553 | 0.6320 | 0.6634 |
| KNN | 0.9823 | 0.9626 | 0.9468 | 0.9546 | 0.9939 | 0.9436 | 0.9437 |
| Random Forest | 0.9883 | 0.9610 | 0.9803 | 0.9705 | 0.9986 | 0.9632 | 0.9633 |
| LightGBM | 0.9868 | 0.9593 | 0.9741 | 0.9666 | 0.9986 | 0.9584 | 0.9584 |
| **Ensemble** | **0.9884** | **0.9627** | **0.9789** | **0.9707** | **0.9990** | **0.9635** | **0.9635** |

The stacking ensemble gives the best accuracy, F1, ROC-AUC, Kappa and MCC,
combining Random Forest's recall with KNN's precision characteristics.

## Repository layout

| Path | Contents |
|---|---|
| `CICIDS-2017/ensemble.py` | **Main framework** — full pipeline, all models, SHAP/PDP/ICE, Sankey |
| `CICIDS-2017/images/` | Generated figures: metrics, confusion matrices, ROC, SHAP, PDP |
| `CICIDS-2017/*_final.py` | Per-model baseline scripts (upstream, see Attribution) |
| `NSL-KDD/`, `RoEduNet-SIMARGL2021/` | Baseline scripts for other datasets (upstream) |
| `Framework/` | Shared preprocessing and SHAP/LIME helpers (upstream) |
| `dev trial/t1/` | Earlier experimental run outputs |

## Getting the dataset

The CICIDS2017 CSVs are **not included** — they are ~1.3 GB, far past
GitHub's limits. Download them from the Canadian Institute for Cybersecurity:

<https://www.unb.ca/cic/datasets/ids-2017.html>

Place the `*.pcap_ISCX.csv` files in `CICIDS-2017/cicids_db/`:

```
CICIDS-2017/cicids_db/
  Monday-WorkingHours.pcap_ISCX.csv
  Tuesday-WorkingHours.pcap_ISCX.csv
  Wednesday-workingHours.pcap_ISCX.csv
  Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
  Friday-WorkingHours-Morning.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Trained models (`*.joblib`) are likewise excluded — the fitted ensemble is
~1.6 GB. Re-running the pipeline regenerates them.

## Running it

```bash
pip install numpy pandas scikit-learn lightgbm shap joblib matplotlib plotly
cd CICIDS-2017
python ensemble.py
```

Everything is written to `CICIDS-2017/images/` (`metrics/`, `roc/`, `shap/`,
`pdp/`, `sankey/`, `instance/`, `models/`).

Key knobs at the top of `ensemble.py`:

| Constant | Default | Meaning |
|---|---|---|
| `N_FEATURES` | 15 | Features kept by variance ranking |
| `TOP_PDP` | 5 | Features plotted in the PDP/ICE panel |
| `TEST_SIZE` | 0.2 | Test split fraction |
| `RANDOM_STATE` | 42 | Seed |
| `SHAP_SAMPLE_SIZE` | 1000 | Cap for tree-explainer sampling |
| `PERM_EXPLAINER_SAMPLES` | 200 | Permutation-explainer budget (KNN) |

## Explainability

- **SHAP** — `TreeExplainer` for Random Forest and LightGBM,
  `PermutationExplainer` for KNN, `KernelExplainer` for the stacked ensemble.
- **PDP + ICE** — `kind='both'` over the top-5 features by importance,
  showing both the average effect and per-instance variability.
- **Sankey** — mean absolute SHAP flow from features to the Attack class.

Top-ranked features are dominated by flow-timing and header attributes:
`Fwd IAT Mean`, `Flow Duration`, `Flow Bytes/s`, `Fwd IAT Std`,
`Fwd Header Length`.

## Known deviations from the write-up

Documented honestly so results can be reproduced and interpreted correctly:

1. **AdaBoost is not in the stack.** `StackingClassifier` is built from
   Random Forest, LightGBM and KNN only. AdaBoost is trained purely as a
   comparison baseline.
2. **The meta-learner is Logistic Regression**, not linear regression —
   appropriate for a classification target.
3. **The train/test split is random-stratified, not time-aware.** CICIDS2017
   has strong temporal structure (Monday is benign-only; attacks cluster by
   day), so a day-based split would be a stricter evaluation.
4. **Feature selection ranks by highest variance and keeps the top 15**,
   rather than applying a near-zero-variance threshold. It runs on unscaled
   data, so large-magnitude timing features are favoured.
5. **The scaler is fit before the split**, which lets test-set statistics
   influence normalisation. Low impact for `StandardScaler`, but worth
   correcting for a strict evaluation.

## Attribution

The per-dataset baseline scripts (`CICIDS-2017/*_final.py`, `NSL-KDD/`,
`RoEduNet-SIMARGL2021/`, `Framework/`, `RF_LIME_SHAP.ipynb`) originate from
the **[ogarreche/XAI_NIDS](https://github.com/ogarreche/XAI_NIDS)** research
repository and remain the work of their original authors.

The contribution in this repository is the **binary stacking-ensemble
framework** — `CICIDS-2017/ensemble.py` — together with its generated
figures and results.
