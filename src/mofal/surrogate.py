"""ExtraTrees surrogate ensemble, uncertainty and out-of-distribution flags."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .features import FEATURE_SET_C

SEEDS = (42, 123, 456, 789, 2024)
ET_PARAMS = dict(n_estimators=300, max_features="sqrt", min_samples_leaf=2)
OOD_DESCRIPTORS = ("density", "PLD_A", "n_atoms", "metal_frac", "C_per_atom")
OOD_QUANTILES = (0.05, 0.95)


@dataclass
class Ensemble:
    models: List[Pipeline]
    features: Sequence[str]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return mu, sigma_seed (std of the M member means) and sigma_tree
        (std over all individual trees of all members, i.e. the total predictive
        spread of the ensemble)."""
        Xf = X[list(self.features)]
        member_means, tree_preds = [], []
        for m in self.models:
            Xi = m.named_steps["imp"].transform(Xf)
            trees = np.stack([t.predict(Xi) for t in m.named_steps["m"].estimators_])
            tree_preds.append(trees)
            member_means.append(trees.mean(axis=0))
        member_means = np.stack(member_means)
        all_trees = np.concatenate(tree_preds, axis=0)
        return pd.DataFrame({
            "mu": member_means.mean(axis=0),
            "sigma_seed": member_means.std(axis=0),
            "sigma_tree": all_trees.std(axis=0),
        }, index=X.index)


def train_ensemble(train: pd.DataFrame, target: str = "gap_PBE",
                   features: Sequence[str] = FEATURE_SET_C, seeds=SEEDS,
                   n_jobs: int = -1, sample_weight=None) -> Ensemble:
    models = []
    for s in seeds:
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", ExtraTreesRegressor(random_state=s, n_jobs=n_jobs, **ET_PARAMS))])
        fit_kw = {"m__sample_weight": sample_weight} if sample_weight is not None else {}
        pipe.fit(train[list(features)], train[target], **fit_kw)
        models.append(pipe)
    return Ensemble(models=models, features=list(features))


def ood_ranges(train: pd.DataFrame, cols=OOD_DESCRIPTORS, q=OOD_QUANTILES) -> Dict[str, tuple]:
    return {c: (float(train[c].quantile(q[0])), float(train[c].quantile(q[1]))) for c in cols}


def flag_ood(df: pd.DataFrame, ranges: Dict[str, tuple]) -> pd.DataFrame:
    reasons = []
    for _, r in df.iterrows():
        rs = [f"{c}<{lo:.2f}" if r[c] < lo else f"{c}>{hi:.2f}"
              for c, (lo, hi) in ranges.items() if pd.notna(r[c]) and not (lo <= r[c] <= hi)]
        reasons.append(";".join(rs))
    out = pd.DataFrame({"ood_reasons": reasons}, index=df.index)
    out["is_ood"] = out.ood_reasons.str.len() > 0
    return out
