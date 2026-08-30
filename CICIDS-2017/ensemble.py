#!/usr/bin/env python3


import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, auc,
    cohen_kappa_score,
    matthews_corrcoef
)

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import joblib

# SHAP / PDP
import shap
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression

# Plotly optional for sankey
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# ---------------- CONFIG ----------------
DATA_DIR = Path("cicids_db")
IMAGES_DIR = Path("images")
SHAP_SAMPLE_SIZE = 1000       # cap for background/data sampling used for tree explainers (smaller -> faster)
PERM_EXPLAINER_SAMPLES = 200  # PermutationExplainer fewer samples to keep KNN feasible
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FEATURES = 15
TOP_PDP = 5
os.makedirs(IMAGES_DIR, exist_ok=True)

# subfolders
for sub in ["metrics", "shap", "pdp", "sankey", "instance", "models", "roc"]:
    (IMAGES_DIR / sub).mkdir(parents=True, exist_ok=True)

# ---------------- Utils ----------------
def robust_read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        print(f"  -> UTF-8 failed, retrying with ISO-8859-1 for {path.name}")
        df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
    return df

def load_and_concat_cicids(data_dir: Path) -> pd.DataFrame:
    csvs = sorted(list(data_dir.glob("*.csv")))
    if not csvs:
        raise RuntimeError(f"No CSV files found in {data_dir}")
    dfs = []
    for f in csvs:
        print("Loading:", f.name)
        try:
            df = robust_read_csv(f)
            df.columns = df.columns.astype(str).str.strip()
            dfs.append(df)
        except Exception as e:
            print("Failed to load", f, e)
            continue
    if not dfs:
        raise RuntimeError("No files loaded.")
    df = pd.concat(dfs, ignore_index=True)
    print("Loaded rows:", len(df))
    return df

def basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    # keep possible label columns; drop other object columns (common in CICIDS they are numeric)
    label_candidates = [c for c in df.columns if c.lower() in ('label','classification','class')]
    to_drop = [c for c in df.columns if df[c].dtype == 'O' and (c not in label_candidates)]
    if to_drop:
        df.drop(columns=to_drop, inplace=True, errors='ignore')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    before = len(df)
    # For simplicity drop any rows with NaN (deterministic). If you want imputation, modify here.
    df.dropna(axis=0, how='any', inplace=True)
    after = len(df)
    print(f"After clean rows: {after} (dropped {before-after})")
    return df

def make_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    # try to find label-like column
    label_col = None
    for c in df.columns:
        if c.lower().strip() in ('label','classification','class'):
            label_col = c
            break
    if label_col is None:
        # fallback: detect 'BENIGN' in any column
        for c in df.columns:
            vals = df[c].dropna().astype(str).str.upper()
            if (vals == 'BENIGN').any():
                label_col = c
                break
    if label_col is None:
        raise RuntimeError(f"No label column found. Available columns: {df.columns}")
    print(f"Found label column: '{label_col}'")
    lab = df[label_col].astype(str).str.strip()
    is_benign = lab.str.upper() == 'BENIGN'
    df['Label_binary'] = (~is_benign).astype(int)  # 0=BENIGN, 1=ATTACK
    return df

def select_features(df: pd.DataFrame, n_features=N_FEATURES, ignore_cols=None) -> List[str]:
    if ignore_cols is None:
        ignore_cols = []
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in ignore_cols]
    if len(numeric) <= n_features:
        return numeric
    variances = df[numeric].var().sort_values(ascending=False)
    return variances.index[:n_features].tolist()

def sanitize_fn(s: str) -> str:
    return "".join([c if c.isalnum() or c in (' ','.','_','-') else '_' for c in s]).replace(' ','_')

# ---------------- Plot helpers ----------------
def save_confusion_matrix(cm, classes, filename: Path, title="Confusion matrix"):
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:,}", horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename, dpi=150)
    plt.close()
    print("Saved confusion matrix:", filename)

def save_bar_metrics(metrics: Dict[str, Dict[str,float]], filename: Path, title="Metrics comparison"):
    labels = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())
    x = np.arange(len(labels))
    width = 0.18
    plt.figure(figsize=(max(8, len(labels)*1.4), 5))
    for i, mname in enumerate(metric_names):
        ys = [metrics[l][mname] for l in labels]
        plt.bar(x + (i - (len(metric_names)-1)/2)*width, ys, width=width, label=mname)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylim(0,1.02)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print("Saved metrics bar chart:", filename)

def save_roc_curves(rocs: Dict[str, tuple], filename: Path, title="ROC Curves"):
    plt.figure(figsize=(7,6))
    for name, (fpr,tpr,roc_auc) in rocs.items():
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.4f})')
    plt.plot([0,1],[0,1], '--', color='grey')
    plt.xlim(0,1); plt.ylim(0,1.05)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(title)
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(filename, dpi=150); plt.close()
    print("Saved ROC curves:", filename)

def metrics_table_png(metrics: Dict[str, Dict[str,float]], filename: Path):
    # render metrics dict into pandas table and save as PNG
    df = pd.DataFrame.from_dict(metrics, orient='index')
    df = df[['accuracy','precision','recall','f1','roc_auc','kappa','mcc']] if 'accuracy' in df.columns else df
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1]*1.2), max(2, df.shape[0]*0.4)))
    ax.axis('off')
    tbl = ax.table(cellText=np.round(df.values,4).astype(str), colLabels=df.columns, rowLabels=df.index, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1,1.2)
    plt.title("Metrics Summary (per model)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print("Saved metrics table PNG:", filename)

# ---------------- Train/Eval ----------------
def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names: List[str], instance_index:int=0):
    models = {}
    # models
    models['AdaBoost'] = AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=50)
    models['KNN'] = KNeighborsClassifier(n_neighbors=7)
    models['RandomForest'] = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    models['LightGBM'] = lgb.LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE)

    # fit
    for name, m in models.items():
        t0 = time.time()
        m.fit(X_train, y_train)
        print(f"Trained {name} in {time.time()-t0:.1f}s")

    print("Training Stacking Ensemble...")

    stack = StackingClassifier(
        estimators=[
            ("rf", models["RandomForest"]),
            ("lgbm", models["LightGBM"]),
            ("knn", models["KNN"])
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1
    )

    stack.fit(X_train, y_train)
    models["Ensemble"] = stack


    per_model_metrics = {}
    rocs = {}
    classes_sorted = np.unique(y_test)
    for name, m in models.items():
        try:
            if hasattr(m, "predict_proba"):
                y_prob = m.predict_proba(X_test)
                # binary case: probability of positive class
                if y_prob.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_prob[:,1])
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:,1])
                else:
                    roc_auc = 0.0; fpr,tpr = [0],[0]
            else:
                y_prob = None; roc_auc = 0.0; fpr,tpr = [0],[0]
        except Exception:
            y_prob = None; roc_auc = 0.0; fpr,tpr = [0],[0]

        y_pred = m.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # --- NEW ROBUST METRICS ---
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        per_model_metrics[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "kappa": kappa,
            "mcc": mcc
        }

        rocs[name] = (fpr, tpr, roc_auc)

        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix(cm, classes=['BENIGN','ATTACK'], filename=IMAGES_DIR/"metrics"/f"{sanitize_fn(name)}_confusion.png", title=f"{name} Confusion Matrix")

    # Save metrics images
    save_bar_metrics(per_model_metrics, IMAGES_DIR/"metrics"/"models_metrics_comparison.png", title="Model Metrics Comparison (binary)")
    for name, mdict in per_model_metrics.items():
        fig, ax = plt.subplots(figsize=(4,3))
        keys = ['accuracy','precision','recall','f1']
        vals = [mdict[k] for k in keys]
        ax.bar(keys, vals)
        ax.set_ylim(0,1.02)
        ax.set_title(f"{name} metrics")
        for i,v in enumerate(vals):
            ax.text(i, v+0.01, f"{v:.4f}", ha='center', fontsize=8)
        plt.tight_layout()
        fn = IMAGES_DIR/"metrics"/f"{sanitize_fn(name)}_metrics.png"
        plt.savefig(fn, dpi=150); plt.close()
        print("Saved per-model metrics:", fn)

    save_roc_curves(rocs, IMAGES_DIR/"roc"/"all_models_roc.png", title="ROC Curves (binary)")
    metrics_table_png(per_model_metrics, IMAGES_DIR/"metrics"/"metrics_summary.png")

    # instance-level comparison
    try:
        x_instance = X_test.iloc[instance_index:instance_index+1] if hasattr(X_test, "iloc") else X_test[instance_index:instance_index+1]
    except Exception:
        x_instance = X_test[:1]
    instance_probs = {}
    for name, m in models.items():
        try:
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(x_instance)[0]
                prob_pos = float(p[1]) if len(p) > 1 else float(np.max(p))
                pred = int(m.predict(x_instance)[0])
                instance_probs[name] = {"prob": prob_pos, "pred": pred}
            else:
                instance_probs[name] = {"prob": None, "pred": int(m.predict(x_instance)[0])}
        except Exception:
            instance_probs[name] = {"prob": None, "pred": None}

    # save instance comparison plot
    names = list(instance_probs.keys())
    probs = [instance_probs[n]["prob"] if instance_probs[n]["prob"] is not None else 0.0 for n in names]
    preds = [instance_probs[n]["pred"] for n in names]
    plt.figure(figsize=(8,4))
    plt.bar(names, probs)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,1.02)
    for i,v in enumerate(probs):
        plt.text(i, v+0.01, f"{v:.3f}\n{preds[i]}", ha='center', fontsize=8)
    plt.title("Instance-level predicted probability by model")
    plt.tight_layout()
    inst_fn = IMAGES_DIR/"instance"/"instance_model_probability_comparison.png"
    plt.savefig(inst_fn, dpi=150)
    plt.close()
    print("Saved:", inst_fn)

    return models, per_model_metrics, instance_probs

# ---------------- SHAP & PDP & Sankey ----------------
def run_shap_and_pdp(
    models: Dict[str, object],
    X_train_scaled: pd.DataFrame,
    X_train_unscaled: pd.DataFrame,
    scaler: StandardScaler,
    feature_names: List[str],
    instance_index=0,
    top_pdp=TOP_PDP
):
    """
    Runs SHAP explanations for:
      - Base models (RF, LightGBM, KNN)
      - Overall stacking ensemble (Kernel SHAP, once)
    Also generates Sankey + PDP.
    """

    os.makedirs(IMAGES_DIR/"shap", exist_ok=True)
    os.makedirs(IMAGES_DIR/"pdp", exist_ok=True)
    os.makedirs(IMAGES_DIR/"sankey", exist_ok=True)

    print("\n=== SHAP EXPLANATION STAGE ===")

    # ---------------- SHAP BACKGROUND & DATA ----------------
    bg = X_train_scaled.sample(
        n=min(100, len(X_train_scaled)),
        random_state=RANDOM_STATE
    )

    data = X_train_scaled.sample(
        n=min(200, len(X_train_scaled)),
        random_state=RANDOM_STATE
    )

    shap_results = {}  # for Sankey (base models only)

    # ================= BASE MODEL SHAP =================
    for name, model in models.items():

        # -------- Tree models --------
        if name in ("RandomForest", "LightGBM"):
            print(f"SHAP for {name} (TreeExplainer)")

            expl = shap.TreeExplainer(
                model,
                data=bg,
                feature_names=feature_names,
                check_additivity=False
            )

            shap_values = expl.shap_values(data)
            arr = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap_results[name] = arr

            # Summary
            shap.summary_plot(
                shap_values,
                data,
                feature_names=feature_names,
                show=False
            )
            plt.savefig(IMAGES_DIR/"shap"/f"{name}_SHAP_Summary.png", dpi=150)
            plt.close()

            # Bar
            shap.summary_plot(
                shap_values,
                data,
                feature_names=feature_names,
                plot_type="bar",
                show=False
            )
            plt.savefig(IMAGES_DIR/"shap"/f"{name}_SHAP_Bar.png", dpi=150)
            plt.close()

        # -------- KNN --------
        elif name == "KNN":
            print("SHAP for KNN (PermutationExplainer)")

            perm = shap.PermutationExplainer(
                model.predict_proba,
                bg
            )

            shap_res = perm(
                data,
                max_evals=PERM_EXPLAINER_SAMPLES
            )

            arr = shap_res.values[:, :, 1]
            shap_results[name] = arr

            shap.summary_plot(
                shap_res,
                data,
                show=False
            )
            plt.savefig(IMAGES_DIR/"shap"/"KNN_SHAP_Summary.png", dpi=150)
            plt.close()

        # -------- Skip unsupported --------
        elif name == "AdaBoost":
            print("Skipping SHAP for AdaBoost (unsupported)")
            continue

    # ================= ENSEMBLE SHAP (ONCE) =================
    print("SHAP for Overall Ensemble (KernelExplainer)")

    def ensemble_predict(X):
        return models["Ensemble"].predict_proba(X)[:, 1]

    bg_e = X_train_scaled.sample(n=50, random_state=RANDOM_STATE)
    data_e = X_train_scaled.sample(n=150, random_state=RANDOM_STATE)

    expl_e = shap.KernelExplainer(
        ensemble_predict,
        bg_e,
        link="logit"
    )

    shap_vals_e = expl_e.shap_values(
        data_e,
        nsamples=200
    )

    # Ensemble summary
    shap.summary_plot(
        shap_vals_e,
        data_e,
        feature_names=feature_names,
        show=False
    )
    plt.savefig(IMAGES_DIR/"shap"/"Ensemble_SHAP_Summary.png", dpi=150)
    plt.close()

    # Ensemble bar
    shap.summary_plot(
        shap_vals_e,
        data_e,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.savefig(IMAGES_DIR/"shap"/"Ensemble_SHAP_Bar.png", dpi=150)
    plt.close()

    print("Saved SHAP for Overall Ensemble")

    # ================= SANKEY (BASE MODELS ONLY) =================
    try:
        if shap_results:
            mean_abs = np.zeros(len(feature_names))
            for arr in shap_results.values():
                mean_abs += np.mean(np.abs(arr), axis=0)
            mean_abs /= len(shap_results)

            labels = list(feature_names) + ["Attack"]
            source = list(range(len(feature_names)))
            target = [len(feature_names)] * len(feature_names)
            value = mean_abs.tolist()

            if _HAS_PLOTLY:
                fig = go.Figure(
                    data=[go.Sankey(
                        node=dict(label=labels),
                        link=dict(source=source, target=target, value=value)
                    )]
                )
                fig.write_html(IMAGES_DIR/"sankey"/"sankey.html")
                try:
                    fig.write_image(IMAGES_DIR/"sankey"/"sankey.png")
                except Exception:
                    pass
            else:
                plt.figure(figsize=(8, 4))
                plt.text(
                    0.5, 0.5,
                    "Install plotly + kaleido for Sankey",
                    ha="center", va="center"
                )
                plt.axis("off")
                plt.savefig(IMAGES_DIR/"sankey"/"sankey_placeholder.png", dpi=150)
                plt.close()

    except Exception as e:
        print("Sankey generation failed:", e)

    # ================= PDP =================
    try:
        compute_and_save_pdp_combined(
            models,
            X_train_scaled,
            feature_names,
            top_k=top_pdp
        )
    except Exception as e:
        print("PDP generation failed:", e)

    print("=== SHAP STAGE COMPLETE ===")


def compute_and_save_pdp_combined(models: Dict[str, object], X_train_scaled: pd.DataFrame, feature_names: List[str], top_k=TOP_PDP):
    """
    Compute PDP & ICE for top_k features (by feature importance if available) and save a single pdp_all.png
    Note: PDP is computed on scaled input (matching model training).
    """
    os.makedirs(IMAGES_DIR/"pdp", exist_ok=True)
    # choose model for PDP (prefer LightGBM then RandomForest)
    model = models.get("LightGBM") or models.get("RandomForest") or next(iter(models.values()))
    # top features by importance if available
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            idx = np.argsort(importances)[::-1][:top_k]
            top_feats = [feature_names[i] for i in idx]
        else:
            top_feats = feature_names[:top_k]
    except Exception:
        top_feats = feature_names[:top_k]

    n = len(top_feats)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4*n))
    if n == 1:
        axes = [axes]
    for ax, feat in zip(axes, top_feats):
        # use PartialDependenceDisplay.from_estimator
        try:
            PartialDependenceDisplay.from_estimator(model, X_train_scaled, [feat], kind='both', ax=ax, ice_lines_kw={'alpha':0.08}, pd_line_kw={'lw':2,'ls':'--'})
            ax.set_title(f"PDP & ICE — {feat}")
        except Exception as e:
            ax.text(0.5,0.5, f"PDP failed for {feat}: {e}", ha='center')
    plt.tight_layout()
    out = IMAGES_DIR/"pdp"/"pdp_all.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved PDP combined:", out)

# ---------------- Main ----------------
def main():
    try:
        df = load_and_concat_cicids(DATA_DIR)
        df = basic_cleanup(df)
        df = make_binary_label(df)
    except Exception as e:
        print("Data load/label fail:", e)
        traceback.print_exc()
        sys.exit(1)

    target_col = 'Label_binary'
    features = select_features(df.drop(columns=[target_col], errors='ignore'), n_features=N_FEATURES)
    if not features:
        print("No numeric features found.")
        sys.exit(1)
    print("Selected features:", features)

    X = df[features]
    y = df[target_col].astype(int)

    # Keep unscaled copy (if needed)
    X_unscaled = X.copy()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    models, per_model_metrics, instance_probs = train_and_evaluate(X_train, y_train, X_test, y_test, feature_names=features, instance_index=0)

    # save models
    for name, m in models.items():
        outp = IMAGES_DIR/"models"/f"{sanitize_fn(name)}.joblib"
        try:
            joblib.dump(m, outp)
            print("Saved trained model:", outp)
        except Exception as e:
            print("Model save failed:", name, e)

    # SHAP + PDP + Sankey
    try:
        run_shap_and_pdp(models, X_train, X_unscaled, scaler, feature_names=features, instance_index=0, top_pdp=TOP_PDP)
    except Exception as e:
        print("SHAP/PDP/Sankey failed:", e)
        traceback.print_exc()

    print("All done. Outputs in 'images/'.")

if __name__ == "__main__":
    main()
