#!/usr/bin/env python3
"""Benchmark of 11 tabular models x 5 feature sets on QMOF PBE band gaps.

Regression target: PBE band gap (eV). Classification target: gap < 1 eV.
Split: 80/20, stratified by the classification label, random_state=42.

Usage
-----
    python scripts/00_qmof_benchmark.py --qmof data/external/qmof.csv --out results/benchmark
    # quick test: --sets C_geometry_composition_elements --models ExtraTrees Ridge

Outputs
-------
    benchmark_regression.csv, benchmark_classification.csv, benchmark_combined.csv
    table_S1_feature_set_C.csv (the table reported in the SI)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, mean_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score, root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mofal.features import FEATURE_SETS, featurize_qmof  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
THRESHOLD = 1.0
N_JOBS = -1


def regressors():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    return {
        "Ridge": (Ridge(alpha=1.0), True),
        "ElasticNet": (ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=20000), True),
        "KNN": (KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=N_JOBS), True),
        "SVR_RBF": (SVR(C=10.0, gamma="scale", epsilon=0.1), True),
        "RandomForest": (RandomForestRegressor(n_estimators=500, max_features="sqrt", n_jobs=N_JOBS, random_state=SEED), False),
        "ExtraTrees": (ExtraTreesRegressor(n_estimators=500, max_features="sqrt", min_samples_leaf=1, n_jobs=N_JOBS, random_state=SEED), False),
        "GradientBoosting": (GradientBoostingRegressor(random_state=SEED), False),
        "HistGradientBoosting": (HistGradientBoostingRegressor(random_state=SEED), False),
        "MLP": (MLPRegressor(hidden_layer_sizes=(256, 128), early_stopping=True, max_iter=500, random_state=SEED), True),
        "XGBoost": (XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, n_jobs=N_JOBS, random_state=SEED), False),
        "LightGBM": (LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, n_jobs=N_JOBS, random_state=SEED, verbose=-1), False),
        "DummyMean": (DummyRegressor(), False),
    }


def classifiers():
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
        "LogisticRegression": (LogisticRegression(max_iter=5000, class_weight="balanced"), True),
        "KNN": (KNeighborsClassifier(n_neighbors=10, weights="distance", n_jobs=N_JOBS), True),
        "SVC_RBF": (SVC(C=10.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED), True),
        "RandomForest": (RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", n_jobs=N_JOBS, random_state=SEED), False),
        "ExtraTrees": (ExtraTreesClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", n_jobs=N_JOBS, random_state=SEED), False),
        "GradientBoosting": (GradientBoostingClassifier(random_state=SEED), False),
        "HistGradientBoosting": (HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED), False),
        "MLP": (MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, max_iter=500, random_state=SEED), True),
        "XGBoost": (XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, n_jobs=N_JOBS, random_state=SEED), False),
        "LightGBM": (LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", n_jobs=N_JOBS, random_state=SEED, verbose=-1), False),
        "DummyMostFrequent": (DummyClassifier(strategy="most_frequent"), False),
    }


def make_pipe(model, numeric, categorical, scale):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric)]
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
        ]), categorical))
    return Pipeline([("prep", ColumnTransformer(transformers)), ("m", model)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmof", required=True)
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument("--sets", nargs="*", default=list(FEATURE_SETS))
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = featurize_qmof(pd.read_csv(args.qmof, low_memory=False))
    print(f"QMOF rows after cleaning: {len(df):,}")
    y = df["gap_PBE"].values
    y_cls = (y < THRESHOLD).astype(int)
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2,
                                      random_state=SEED, stratify=y_cls)
    pd.Series(df.iloc[idx_te]["qmof_id"].values).to_csv(out / "test_ids.csv", index=False, header=["qmof_id"])
    print(f"train {len(idx_tr):,}  test {len(idx_te):,}")

    # resume support: keep rows of feature sets that are not re-run
    reg_rows, cls_rows = [], []
    for fname, rows in (("benchmark_regression.csv", reg_rows), ("benchmark_classification.csv", cls_rows)):
        if (out / fname).exists():
            old = pd.read_csv(out / fname)
            rows.extend(old[~old.feature_set.isin(args.sets)].to_dict("records"))
    for fs in args.sets:
        numeric, categorical = FEATURE_SETS[fs]
        X = df[numeric + categorical]
        Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
        for name, (model, scale) in regressors().items():
            if args.models and name not in args.models:
                continue
            t0 = time.time()
            pipe = make_pipe(model, numeric, categorical, scale)
            pipe.fit(Xtr, y[idx_tr])
            p = pipe.predict(Xte)
            yt = y[idx_te]
            reg_rows.append(dict(task="regression", feature_set=fs, model=name,
                                 n_train=len(idx_tr), n_test=len(idx_te),
                                 n_numeric=len(numeric), n_categorical=len(categorical),
                                 time_sec=round(time.time() - t0, 1),
                                 R2=r2_score(yt, p), MAE_eV=mean_absolute_error(yt, p),
                                 RMSE_eV=root_mean_squared_error(yt, p),
                                 Spearman=spearmanr(yt, p).statistic if name != "DummyMean" else np.nan))
            print(f"[reg] {fs[:1]} {name:22s} R2={reg_rows[-1]['R2']:.3f} MAE={reg_rows[-1]['MAE_eV']:.3f}", flush=True)
        for name, (model, scale) in classifiers().items():
            if args.models and name not in args.models:
                continue
            t0 = time.time()
            pipe = make_pipe(model, numeric, categorical, scale)
            pipe.fit(Xtr, y_cls[idx_tr])
            prob = pipe.predict_proba(Xte)[:, 1]
            pred = (prob >= 0.5).astype(int)
            yt = y_cls[idx_te]
            cls_rows.append(dict(task="classification", feature_set=fs, model=name,
                                 n_train=len(idx_tr), n_test=len(idx_te),
                                 time_sec=round(time.time() - t0, 1), threshold_eV=THRESHOLD,
                                 Accuracy=accuracy_score(yt, pred),
                                 Balanced_accuracy=balanced_accuracy_score(yt, pred),
                                 F1=f1_score(yt, pred, zero_division=0),
                                 Precision=precision_score(yt, pred, zero_division=0),
                                 Recall=recall_score(yt, pred, zero_division=0),
                                 ROC_AUC=roc_auc_score(yt, prob), PR_AUC=average_precision_score(yt, prob)))
            print(f"[cls] {fs[:1]} {name:22s} AUC={cls_rows[-1]['ROC_AUC']:.3f} F1={cls_rows[-1]['F1']:.3f}", flush=True)
        pd.DataFrame(reg_rows).to_csv(out / "benchmark_regression.csv", index=False)
        pd.DataFrame(cls_rows).to_csv(out / "benchmark_classification.csv", index=False)

    reg, cls = pd.DataFrame(reg_rows), pd.DataFrame(cls_rows)
    pd.concat([reg, cls]).to_csv(out / "benchmark_combined.csv", index=False)
    # Table S1: feature set C; classifier paired by model family
    pair = {"Ridge": "LogisticRegression", "ElasticNet": "LogisticRegression",
            "SVR_RBF": "SVC_RBF", "DummyMean": "DummyMostFrequent"}
    c = reg[reg.feature_set == "C_geometry_composition_elements"].copy()
    if len(c):
        cc = cls[cls.feature_set == "C_geometry_composition_elements"].set_index("model")
        c["classifier_used"] = c.model.map(lambda m: pair.get(m, m))
        c["ROC_AUC"] = c.classifier_used.map(cc["ROC_AUC"])
        c["F1"] = c.classifier_used.map(cc["F1"])
        c[["model", "R2", "MAE_eV", "RMSE_eV", "Spearman", "classifier_used", "ROC_AUC", "F1"]] \
            .round(3).to_csv(out / "table_S1_feature_set_C.csv", index=False)


if __name__ == "__main__":
    main()
