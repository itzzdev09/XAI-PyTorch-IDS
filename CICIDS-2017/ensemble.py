#!/usr/bin/env python3
"""
An Explainable Ensemble-Based Framework for Binary Network Intrusion Detection
=============================================================================

Implements the framework described in the accompanying paper:

  Data Preparation
      CICIDS2017 CSVs
        -> cleaning & binary labeling (BENIGN -> 0, any attack -> 1)   [Eq. 2]
        -> feature harmonisation across capture files                  [Sec 3.4]
        -> noise reduction (constant / duplicate / degenerate columns)  [Sec 3.4]
        -> time-sensitive train / test split                            [Sec 2.4]
        -> variance-threshold feature selection  (fit on train only)    [Sec 3.4]
        -> StandardScaler normalisation          (fit on train only)    [Fig. 1]

  Level-0 base learners                                          [Table 1]
        Random Forest (bagging)      -> variance reduction
        LightGBM      (grad. boost)  -> bias reduction
        AdaBoost      (adaptive)     -> error reweighting
        KNN           (instance)     -> geometric structure

  Level-1 meta-learner                                        [Sec 3.3, Eq. 5]
        Logistic Regression -> learns coefficients (beta) over the
        base learners' predicted probabilities

  Integrated explainability                                        [Sec 3.5]
        SHAP  (local attribution, per model + whole ensemble)
        PDP   (global effect)  +  ICE (per-instance variability)
        Sankey (mean |SHAP| flow from features to the Attack class)

Outputs are written to images/ (metrics, roc, shap, pdp, sankey, instance,
models).
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve,
    cohen_kappa_score,
    matthews_corrcoef,
)

from sklearn.ensemble import (
    AdaBoostClassifier, RandomForestClassifier, StackingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import joblib

# SHAP / PDP
import shap
from sklearn.inspection import PartialDependenceDisplay

# Plotly optional for sankey
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


# ============================ CONFIG ============================
DATA_DIR    = Path("cicids_db")
IMAGES_DIR  = Path("images")

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_FEATURES   = 15      # features retained after variance selection  (Fig. 3)
TOP_PDP      = 5       # features plotted in the PDP/ICE panel        (Fig. 4)

# Sec. 3.4 - variance filtering. Features whose variance falls below the
# threshold carry no discriminative power and are removed; the survivors are
# then ranked and the top N_FEATURES kept.
#
# VARIANCE_BASIS controls the scale variance is measured on:
#   "raw"    : variance of the untransformed feature. This is what produced
#              the feature set reported in the paper (Fig. 3) -- but it is
#              scale-dependent, so microsecond-scale timing columns dominate
#              simply because of their units. DEFAULT, for reproducibility.
#   "minmax" : each feature is squeezed to [0, 1] first, so variance becomes
#              a unit-free measure of spread. Scale-fair, and the more
#              defensible choice if you are willing to re-run and update the
#              reported feature set.
#
# Note: variance must NOT be measured after StandardScaler -- that forces
# every variance to exactly 1.0 and the ranking becomes meaningless.
VARIANCE_BASIS = "raw"
VARIANCE_THRESHOLD = {"raw": 0.0, "minmax": 1e-4}

# Sec. 2.4 - time-sensitive evaluation scheme.
#   "per_day"     : within each capture day, the earlier flows train and the
#                   later flows test. Chronological (no future -> past leak)
#                   while keeping every attack family represented. DEFAULT.
#   "global_time" : strict chronological split over the whole capture week.
#                   Friday's attacks (Bot / PortScan / DDoS) end up test-only,
#                   so this measures generalisation to UNSEEN attack families
#                   and scores far lower. Use it for a worst-case report.
#   "stratified"  : random stratified split. Not time-sensitive; provided to
#                   reproduce the original Table 2 numbers.
SPLIT_MODE = "per_day"

# SHAP sampling budgets
SHAP_BACKGROUND        = 100  # background set for tree explainers
SHAP_DATA              = 200  # rows explained for tree explainers
PERM_EXPLAINER_SAMPLES = 200  # PermutationExplainer budget (KNN / AdaBoost)
ENSEMBLE_BG            = 50   # KernelExplainer background (ensemble)
ENSEMBLE_DATA          = 150  # rows explained for the ensemble
ENSEMBLE_NSAMPLES      = 200  # KernelExplainer coalitions

# Capture-day ordering of the CICIDS2017 files (Mon 3 Jul -> Fri 7 Jul 2017).
# Used to order days when timestamps are unparseable.
DAY_ORDER = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday-workinghours-morning",
    "thursday-workinghours-afternoon",
    "friday-workinghours-morning",
    "friday-workinghours-afternoon-portscan",
    "friday-workinghours-afternoon-ddos",
]

os.makedirs(IMAGES_DIR, exist_ok=True)
for _sub in ["metrics", "shap", "pdp", "sankey", "instance", "models", "roc"]:
    (IMAGES_DIR / _sub).mkdir(parents=True, exist_ok=True)


# ============================ Utils ============================
def robust_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        print(f"  -> UTF-8 failed, retrying with ISO-8859-1 for {path.name}")
        return pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)


def _day_rank(filename: str) -> int:
    """Chronological rank of a capture file, from its name."""
    low = filename.lower()
    for i, key in enumerate(DAY_ORDER):
        if low.startswith(key):
            return i
    return len(DAY_ORDER)


def parse_cicids_timestamp(series: pd.Series, is_afternoon: bool) -> pd.Series:
    """
    CICIDS2017 timestamps are inconsistent between files:
        Monday          -> '03/07/2017 08:55:58'   (zero padded, with seconds)
        Others          -> '4/7/2017 8:54'         (no padding, no seconds)
        Afternoon files -> '7/7/2017 3:30'         (12-hour clock, no AM/PM)

    Parsed day-first. In afternoon captures an hour < 8 is really PM, so 12
    hours are added -- the captures run roughly 08:00-17:00.
    """
    ts = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if is_afternoon:
        need_pm = ts.dt.hour < 8
        ts = ts.where(~need_pm, ts + pd.Timedelta(hours=12))
    return ts


def load_and_concat_cicids(data_dir: Path) -> pd.DataFrame:
    """
    Load every capture file, harmonise the feature space across them
    (Sec. 3.4) and attach the capture-day / timestamp columns needed by the
    time-sensitive split (Sec. 2.4).
    """
    csvs = sorted(data_dir.glob("*.csv"), key=lambda p: _day_rank(p.name))
    if not csvs:
        raise RuntimeError(f"No CSV files found in {data_dir}")

    frames, col_sets = [], []
    for f in csvs:
        print("Loading:", f.name)
        try:
            df = robust_read_csv(f)
        except Exception as e:
            print("  !! failed to load", f.name, e)
            continue
        df.columns = df.columns.astype(str).str.strip()

        rank = _day_rank(f.name)
        is_pm = "afternoon" in f.name.lower()
        df["__day_rank"] = rank
        df["__source"] = f.name
        if "Timestamp" in df.columns:
            df["__ts"] = parse_cicids_timestamp(df["Timestamp"], is_pm)
        else:
            df["__ts"] = pd.NaT

        frames.append(df)
        col_sets.append(set(df.columns))

    if not frames:
        raise RuntimeError("No files loaded.")

    # ---- Feature harmonisation (Sec. 3.4) -------------------------------
    # Keep only the columns present in EVERY capture file, so each classifier
    # operates in one consistent feature space across capture times.
    common = set.intersection(*col_sets)
    dropped = sorted(set.union(*col_sets) - common)
    if dropped:
        print(f"Harmonisation: dropping {len(dropped)} column(s) not common "
              f"to all files: {dropped[:8]}{' ...' if len(dropped) > 8 else ''}")
    ordered = [c for c in frames[0].columns if c in common]
    frames = [f[ordered] for f in frames]

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded rows: {len(df):,}  |  columns: {df.shape[1]}")
    return df


def basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning + noise reduction (Sec. 3.4)."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    keep = {"__day_rank", "__source", "__ts"}
    label_candidates = [c for c in df.columns
                        if c.lower() in ("label", "classification", "class")]

    # Drop identifier / free-text object columns (Flow ID, IPs, Timestamp...).
    # They are not features and several leak host identity directly.
    to_drop = [c for c in df.columns
               if df[c].dtype == "O" and c not in label_candidates and c not in keep]
    if to_drop:
        print(f"Dropping {len(to_drop)} non-feature object column(s): {to_drop}")
        df.drop(columns=to_drop, inplace=True, errors="ignore")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(df)
    df.dropna(axis=0, how="all", inplace=True)      # blank padding rows
    df.dropna(axis=0, how="any", inplace=True)      # residual NaN / inf rows
    print(f"After clean rows: {len(df):,} (dropped {before - len(df):,})")
    return df


def make_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Dichotomous abstraction, Eq. 2: BENIGN -> 0, every attack -> 1."""
    label_col = None
    for c in df.columns:
        if c.lower().strip() in ("label", "classification", "class"):
            label_col = c
            break
    if label_col is None:
        for c in df.columns:
            vals = df[c].dropna().astype(str).str.upper()
            if (vals == "BENIGN").any():
                label_col = c
                break
    if label_col is None:
        raise RuntimeError(f"No label column found. Columns: {list(df.columns)}")

    print(f"Found label column: '{label_col}'")
    lab = df[label_col].astype(str).str.strip()
    df["Label_binary"] = (lab.str.upper() != "BENIGN").astype(int)

    n_att = int(df["Label_binary"].sum())
    print(f"Binary labels -> BENIGN {len(df) - n_att:,} | ATTACK {n_att:,} "
          f"({100 * n_att / max(len(df), 1):.2f}% attack)")
    df.drop(columns=[label_col], inplace=True)
    return df


def reduce_noise(X: pd.DataFrame) -> pd.DataFrame:
    """Noise reduction (Sec. 3.4): drop constant and duplicated columns."""
    const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const:
        print(f"Noise reduction: dropping {len(const)} constant column(s)")
        X = X.drop(columns=const)
    dup = X.columns[X.T.duplicated()].tolist()
    if dup:
        print(f"Noise reduction: dropping {len(dup)} duplicated column(s): {dup}")
        X = X.drop(columns=dup)
    return X


# ==================== Time-sensitive split (Sec. 2.4) ====================
def time_sensitive_split(df: pd.DataFrame, target_col: str,
                         mode: str = SPLIT_MODE,
                         test_size: float = TEST_SIZE
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Build a temporally ordered train/test split.

    per_day     - within each capture day, the earliest (1-test_size) of the
                  flows train and the latest test_size test. Chronological,
                  and every attack family stays represented on both sides.
    global_time - one chronological cut across the whole week. Later attack
                  families become test-only (unseen-attack generalisation).
    stratified  - random stratified split (not time-sensitive; reproduces the
                  original results).
    """
    feat_cols = [c for c in df.columns
                 if c not in (target_col, "__day_rank", "__source", "__ts")]
    y_all = df[target_col].astype(int)

    if mode == "stratified":
        print("Split: random stratified (NOT time-sensitive)")
        return train_test_split(df[feat_cols], y_all, test_size=test_size,
                                random_state=RANDOM_STATE, stratify=y_all)

    # Order chronologically: capture day first, timestamp within the day.
    # Rows with unparseable timestamps keep their original file order.
    order = df.sort_values(
        by=["__day_rank", "__ts"], kind="mergesort", na_position="last"
    ).index

    if mode == "global_time":
        cut = int(len(order) * (1 - test_size))
        train_idx, test_idx = order[:cut], order[cut:]
        print(f"Split: strict chronological over the full capture week "
              f"(train {len(train_idx):,} / test {len(test_idx):,})")
    elif mode == "per_day":
        train_parts, test_parts = [], []
        ordered = df.loc[order]
        for rank, chunk in ordered.groupby("__day_rank", sort=True):
            cut = int(len(chunk) * (1 - test_size))
            train_parts.append(chunk.index[:cut])
            test_parts.append(chunk.index[cut:])
            src = chunk["__source"].iloc[0]
            print(f"  day {rank} ({src[:42]:42s}) "
                  f"train {cut:>7,} | test {len(chunk) - cut:>7,}")
        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        print(f"Split: per-day chronological "
              f"(train {len(train_idx):,} / test {len(test_idx):,})")
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {mode}")

    X_train = df.loc[train_idx, feat_cols]
    X_test = df.loc[test_idx, feat_cols]
    y_train = y_all.loc[train_idx]
    y_test = y_all.loc[test_idx]

    for nm, yy in (("train", y_train), ("test", y_test)):
        n_att = int(yy.sum())
        print(f"    {nm}: BENIGN {len(yy) - n_att:,} | ATTACK {n_att:,} "
              f"({100 * n_att / max(len(yy), 1):.2f}%)")
    if y_test.nunique() < 2 or y_train.nunique() < 2:
        raise RuntimeError(
            "A split fold contains a single class. Try SPLIT_MODE='per_day'."
        )
    return X_train, X_test, y_train, y_test


# ============ Feature selection + scaling (fit on TRAIN only) ============
def select_features_and_scale(X_train: pd.DataFrame, X_test: pd.DataFrame,
                              n_features: int = N_FEATURES
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], StandardScaler]:
    """
    Variance-threshold feature selection (Sec. 3.4 / Fig. 1) followed by
    StandardScaler normalisation.

    Both are fit on the TRAINING fold only, so no test-set statistic can
    influence selection or normalisation.

    Variance is measured on the basis chosen by VARIANCE_BASIS -- never after
    StandardScaler, which would force every variance to 1.0 and make the
    ranking degenerate.
    """
    X_train = X_train.select_dtypes(include=[np.number]).copy()
    X_test = X_test[X_train.columns].copy()

    # Basis on which variance is measured (fit on the training fold only).
    if VARIANCE_BASIS == "minmax":
        basis = MinMaxScaler().fit(X_train)
        V_train = pd.DataFrame(basis.transform(X_train),
                               columns=X_train.columns, index=X_train.index)
    elif VARIANCE_BASIS == "raw":
        V_train = X_train
    else:
        raise ValueError(f"Unknown VARIANCE_BASIS: {VARIANCE_BASIS}")
    thresh = VARIANCE_THRESHOLD[VARIANCE_BASIS]

    # 1) Variance filtering: remove near-zero-variance features.
    vt = VarianceThreshold(threshold=thresh).fit(V_train)
    kept = X_train.columns[vt.get_support()].tolist()
    removed = [c for c in X_train.columns if c not in kept]
    print(f"Variance filter (basis={VARIANCE_BASIS}, threshold={thresh}): "
          f"kept {len(kept)}, removed {len(removed)} near-zero-variance "
          f"feature(s)")
    if removed:
        print(f"    removed: {removed}")

    # 2) Rank the survivors and keep the top N.
    variances = V_train[kept].var().sort_values(ascending=False)
    features = variances.index[:n_features].tolist()
    print(f"Selected {len(features)} feature(s) by variance rank:")
    for f in features:
        print(f"    {f:<38s} var={variances[f]:.6g}")

    # 3) Final scaler on the selected features, fit on train only.
    scaler = StandardScaler().fit(X_train[features])
    X_train_s = pd.DataFrame(scaler.transform(X_train[features]),
                             columns=features, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test[features]),
                            columns=features, index=X_test.index)
    return X_train_s, X_test_s, features, scaler


def sanitize_fn(s: str) -> str:
    return "".join(c if c.isalnum() or c in (" ", ".", "_", "-") else "_"
                   for c in s).replace(" ", "_")


# ============================ Plot helpers ============================
def save_confusion_matrix(cm, classes, filename: Path, title="Confusion matrix"):
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print("Saved confusion matrix:", filename)


def save_bar_metrics(metrics: Dict[str, Dict[str, float]], filename: Path,
                     title="Metrics comparison"):
    labels = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())
    x = np.arange(len(labels))
    width = 0.8 / max(len(metric_names), 1)
    plt.figure(figsize=(max(8, len(labels) * 1.6), 5))
    for i, mname in enumerate(metric_names):
        ys = [metrics[l][mname] for l in labels]
        plt.bar(x + (i - (len(metric_names) - 1) / 2) * width, ys,
                width=width, label=mname)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1.02)
    plt.legend(fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print("Saved metrics bar chart:", filename)


def save_roc_curves(rocs: Dict[str, tuple], filename: Path, title="ROC Curves"):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, roc_auc) in rocs.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print("Saved ROC curves:", filename)


def metrics_table_png(metrics: Dict[str, Dict[str, float]], filename: Path):
    df = pd.DataFrame.from_dict(metrics, orient="index")
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "kappa", "mcc"]
    df = df[[c for c in cols if c in df.columns]]
    fig, ax = plt.subplots(figsize=(max(7, df.shape[1] * 1.3),
                                    max(2, df.shape[0] * 0.45)))
    ax.axis("off")
    tbl = ax.table(cellText=np.round(df.values, 4).astype(str),
                   colLabels=df.columns, rowLabels=df.index, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    plt.title("Metrics Summary (per model)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    df.to_csv(filename.with_suffix(".csv"))
    print("Saved metrics table:", filename, "and", filename.with_suffix(".csv"))


# ============================ Train / Eval ============================
def build_models() -> Dict[str, object]:
    """Level-0 base learners (Table 1)."""
    return {
        "AdaBoost": AdaBoostClassifier(n_estimators=50,
                                       random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                               random_state=RANDOM_STATE),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, verbose=-1,
                                       random_state=RANDOM_STATE),
    }


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       feature_names: List[str], instance_index: int = 0):
    models = build_models()

    for name, m in models.items():
        t0 = time.time()
        m.fit(X_train, y_train)
        print(f"Trained {name} in {time.time() - t0:.1f}s")

    # ---- Stacking ensemble (Sec. 3.2) --------------------------------
    # All FOUR Level-0 learners from Table 1 feed the meta-learner, so the
    # ensemble spans the bagging / boosting / instance-based paradigms the
    # diversity argument in Sec. 2.2 relies on.
    #
    # The meta-learner is LogisticRegression: it applies exactly the linear
    # weighting of Eq. 5 -- an intercept beta_0 plus a coefficient beta_k per
    # base learner -- through a logistic link, which is what a {0,1} target
    # requires. Plain LinearRegression is not a classifier and cannot be used
    # as a StackingClassifier final estimator.
    print("Training Stacking Ensemble (RF + LightGBM + AdaBoost + KNN)...")
    t0 = time.time()
    stack = StackingClassifier(
        estimators=[
            ("rf", models["RandomForest"]),
            ("lgbm", models["LightGBM"]),
            ("ada", models["AdaBoost"]),
            ("knn", models["KNN"]),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        passthrough=False,
        cv=5,
        n_jobs=1,
    )
    stack.fit(X_train, y_train)
    models["Ensemble"] = stack
    print(f"Trained Ensemble in {time.time() - t0:.1f}s")

    # Report the learned meta-learner weights (Eq. 5).
    try:
        coefs = stack.final_estimator_.coef_.ravel()
        names = [n for n, _ in stack.estimators]
        print("Meta-learner coefficients (Eq. 5):")
        print(f"    beta_0 (intercept) = {stack.final_estimator_.intercept_[0]:+.4f}")
        for n, c in zip(names, coefs):
            print(f"    beta[{n:<5s}]        = {c:+.4f}")
        with open(IMAGES_DIR / "metrics" / "meta_learner_coefficients.json", "w") as fh:
            json.dump({"intercept": float(stack.final_estimator_.intercept_[0]),
                       "coefficients": dict(zip(names, map(float, coefs)))},
                      fh, indent=2)
    except Exception as e:
        print("Could not extract meta-learner coefficients:", e)

    # ---- Evaluation --------------------------------------------------
    per_model_metrics, rocs = {}, {}
    for name, m in models.items():
        try:
            y_prob = m.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            else:
                roc_auc, fpr, tpr = 0.0, [0], [0]
        except Exception:
            roc_auc, fpr, tpr = 0.0, [0], [0]

        y_pred = m.predict(X_test)
        per_model_metrics[name] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":   roc_auc,
            "kappa":     cohen_kappa_score(y_test, y_pred),
            "mcc":       matthews_corrcoef(y_test, y_pred),
        }
        rocs[name] = (fpr, tpr, roc_auc)

        save_confusion_matrix(
            confusion_matrix(y_test, y_pred),
            classes=["BENIGN", "ATTACK"],
            filename=IMAGES_DIR / "metrics" / f"{sanitize_fn(name)}_confusion.png",
            title=f"{name} Confusion Matrix",
        )

    save_bar_metrics(per_model_metrics,
                     IMAGES_DIR / "metrics" / "models_metrics_comparison.png",
                     title="Model Metrics Comparison (binary)")
    for name, mdict in per_model_metrics.items():
        fig, ax = plt.subplots(figsize=(4, 3))
        keys = ["accuracy", "precision", "recall", "f1"]
        vals = [mdict[k] for k in keys]
        ax.bar(keys, vals)
        ax.set_ylim(0, 1.10)
        ax.set_title(f"{name} metrics")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
        plt.tight_layout()
        fn = IMAGES_DIR / "metrics" / f"{sanitize_fn(name)}_metrics.png"
        plt.savefig(fn, dpi=150)
        plt.close()
        print("Saved per-model metrics:", fn)

    save_roc_curves(rocs, IMAGES_DIR / "roc" / "all_models_roc.png",
                    title="ROC Curves (binary)")
    metrics_table_png(per_model_metrics,
                      IMAGES_DIR / "metrics" / "metrics_summary.png")

    # ---- Instance-level comparison -----------------------------------
    x_instance = X_test.iloc[instance_index:instance_index + 1]
    instance_probs = {}
    for name, m in models.items():
        try:
            p = m.predict_proba(x_instance)[0]
            instance_probs[name] = {
                "prob": float(p[1]) if len(p) > 1 else float(np.max(p)),
                "pred": int(m.predict(x_instance)[0]),
            }
        except Exception:
            instance_probs[name] = {"prob": None, "pred": None}

    names = list(instance_probs)
    probs = [instance_probs[n]["prob"] or 0.0 for n in names]
    preds = [instance_probs[n]["pred"] for n in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, probs)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.10)
    for i, v in enumerate(probs):
        plt.text(i, v + 0.01, f"{v:.3f}\n{preds[i]}", ha="center", fontsize=8)
    plt.title("Instance-level predicted probability by model")
    plt.tight_layout()
    inst_fn = IMAGES_DIR / "instance" / "instance_model_probability_comparison.png"
    plt.savefig(inst_fn, dpi=150)
    plt.close()
    print("Saved:", inst_fn)

    return models, per_model_metrics, instance_probs


# ==================== SHAP / PDP / ICE / Sankey ====================
def run_shap_and_pdp(models: Dict[str, object], X_train_scaled: pd.DataFrame,
                     feature_names: List[str], top_pdp: int = TOP_PDP):
    """Integrated explainability (Sec. 3.5)."""
    print("\n=== SHAP EXPLANATION STAGE ===")
    bg = X_train_scaled.sample(n=min(SHAP_BACKGROUND, len(X_train_scaled)),
                               random_state=RANDOM_STATE)
    data = X_train_scaled.sample(n=min(SHAP_DATA, len(X_train_scaled)),
                                 random_state=RANDOM_STATE)
    shap_results = {}

    for name, model in models.items():
        if name == "Ensemble":
            continue
        try:
            if name in ("RandomForest", "LightGBM"):
                print(f"SHAP for {name} (TreeExplainer)")
                expl = shap.TreeExplainer(model, data=bg,
                                          feature_names=feature_names,
                                          check_additivity=False)
                sv = expl.shap_values(data)
                arr = sv[1] if isinstance(sv, list) else sv
                if arr.ndim == 3:
                    arr = arr[:, :, 1]
                shap_results[name] = arr
                plot_obj = arr
            else:
                # KNN and AdaBoost expose no tree structure -> permutation.
                # AdaBoost is now a stacked base learner, so it is explained
                # alongside the others rather than skipped.
                print(f"SHAP for {name} (PermutationExplainer)")
                perm = shap.PermutationExplainer(model.predict_proba, bg)
                res = perm(data, max_evals=PERM_EXPLAINER_SAMPLES)
                arr = res.values[:, :, 1]
                shap_results[name] = arr
                plot_obj = arr

            shap.summary_plot(plot_obj, data, feature_names=feature_names,
                              show=False)
            plt.savefig(IMAGES_DIR / "shap" / f"{name}_SHAP_Summary.png", dpi=150)
            plt.close()

            shap.summary_plot(plot_obj, data, feature_names=feature_names,
                              plot_type="bar", show=False)
            plt.savefig(IMAGES_DIR / "shap" / f"{name}_SHAP_Bar.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"  !! SHAP failed for {name}: {e}")

    # ---- Whole-ensemble SHAP ----------------------------------------
    try:
        print("SHAP for Overall Ensemble (KernelExplainer)")
        bg_e = X_train_scaled.sample(n=min(ENSEMBLE_BG, len(X_train_scaled)),
                                     random_state=RANDOM_STATE)
        data_e = X_train_scaled.sample(n=min(ENSEMBLE_DATA, len(X_train_scaled)),
                                       random_state=RANDOM_STATE)
        expl_e = shap.KernelExplainer(
            lambda X: models["Ensemble"].predict_proba(X)[:, 1], bg_e, link="logit")
        sv_e = expl_e.shap_values(data_e, nsamples=ENSEMBLE_NSAMPLES)

        shap.summary_plot(sv_e, data_e, feature_names=feature_names, show=False)
        plt.savefig(IMAGES_DIR / "shap" / "Ensemble_SHAP_Summary.png", dpi=150)
        plt.close()
        shap.summary_plot(sv_e, data_e, feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.savefig(IMAGES_DIR / "shap" / "Ensemble_SHAP_Bar.png", dpi=150)
        plt.close()
        print("Saved SHAP for Overall Ensemble")
    except Exception as e:
        print("Ensemble SHAP failed:", e)

    # ---- Sankey: mean |SHAP| flow feature -> Attack -------------------
    try:
        if shap_results:
            mean_abs = np.zeros(len(feature_names))
            for arr in shap_results.values():
                mean_abs += np.mean(np.abs(arr), axis=0)
            mean_abs /= len(shap_results)

            if _HAS_PLOTLY:
                fig = go.Figure(data=[go.Sankey(
                    node=dict(label=list(feature_names) + ["Attack"]),
                    link=dict(source=list(range(len(feature_names))),
                              target=[len(feature_names)] * len(feature_names),
                              value=mean_abs.tolist()))])
                fig.write_html(IMAGES_DIR / "sankey" / "sankey.html")
                try:
                    fig.write_image(IMAGES_DIR / "sankey" / "sankey.png")
                except Exception:
                    pass
                print("Saved Sankey diagram")
            else:
                print("plotly not installed - skipping Sankey")
    except Exception as e:
        print("Sankey generation failed:", e)

    try:
        compute_and_save_pdp_combined(models, X_train_scaled, feature_names,
                                      top_k=top_pdp)
    except Exception as e:
        print("PDP generation failed:", e)

    print("=== SHAP STAGE COMPLETE ===")


def compute_and_save_pdp_combined(models: Dict[str, object],
                                  X_train_scaled: pd.DataFrame,
                                  feature_names: List[str],
                                  top_k: int = TOP_PDP):
    """PDP (global effect) + ICE (per-instance variability), Sec. 3.5 / Fig. 4."""
    model = models.get("LightGBM") or models.get("RandomForest")
    try:
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_k]
        top_feats = [feature_names[i] for i in idx]
    except Exception:
        top_feats = feature_names[:top_k]

    n = len(top_feats)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n))
    axes = np.atleast_1d(axes)
    for ax, feat in zip(axes, top_feats):
        try:
            PartialDependenceDisplay.from_estimator(
                model, X_train_scaled, [feat], kind="both", ax=ax,
                ice_lines_kw={"alpha": 0.08},
                pd_line_kw={"lw": 2, "ls": "--"})
            ax.set_title(f"PDP & ICE - {feat}")
        except Exception as e:
            ax.text(0.5, 0.5, f"PDP failed for {feat}: {e}", ha="center")
    plt.tight_layout()
    out = IMAGES_DIR / "pdp" / "pdp_all.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved PDP combined:", out)


# ============================ Main ============================
def main():
    print("=" * 70)
    print("Explainable Ensemble Framework for Binary NIDS - CICIDS2017")
    print(f"Split mode: {SPLIT_MODE} | features: {N_FEATURES} | seed: {RANDOM_STATE}")
    print("=" * 70)

    try:
        df = load_and_concat_cicids(DATA_DIR)
        df = basic_cleanup(df)
        df = make_binary_label(df)
    except Exception as e:
        print("Data load/label failed:", e)
        traceback.print_exc()
        sys.exit(1)

    target_col = "Label_binary"
    meta = df[["__day_rank", "__source", "__ts"]]
    feats = reduce_noise(df.drop(columns=[target_col, "__day_rank",
                                          "__source", "__ts"]))
    df = pd.concat([feats, meta, df[target_col]], axis=1)

    # 1) Time-sensitive split FIRST, so nothing downstream sees the test fold.
    X_train_raw, X_test_raw, y_train, y_test = time_sensitive_split(
        df, target_col, mode=SPLIT_MODE, test_size=TEST_SIZE)

    # 2) Variance selection + scaling, both fit on the training fold only.
    X_train, X_test, features, scaler = select_features_and_scale(
        X_train_raw, X_test_raw, n_features=N_FEATURES)

    models, per_model_metrics, instance_probs = train_and_evaluate(
        X_train, y_train, X_test, y_test, feature_names=features,
        instance_index=0)

    joblib.dump(scaler, IMAGES_DIR / "models" / "scaler.joblib")
    for name, m in models.items():
        try:
            joblib.dump(m, IMAGES_DIR / "models" / f"{sanitize_fn(name)}.joblib")
            print("Saved trained model:", name)
        except Exception as e:
            print("Model save failed:", name, e)

    try:
        run_shap_and_pdp(models, X_train, feature_names=features,
                         top_pdp=TOP_PDP)
    except Exception as e:
        print("SHAP/PDP/Sankey failed:", e)
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Final results")
    print("=" * 70)
    print(pd.DataFrame.from_dict(per_model_metrics, orient="index").round(4)
          .to_string())
    print("\nAll done. Outputs in 'images/'.")


if __name__ == "__main__":
    main()
