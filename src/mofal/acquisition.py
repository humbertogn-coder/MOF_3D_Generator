"""Multi-objective lower-confidence-bound acquisition with diversity caps.

score(x) = mu(x) - kappa*sigma(x) - alpha / max(1, N_topo(x)) + beta * I_ood(x)

Lower is better (we look for low band gaps). N_topo counts structures with an
accepted DFT label in the topology of x. Selection walks the ranked list and
enforces at most ``max_per_topology``, ``max_per_metal`` and ``max_per_substituent``
picks per batch.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Optional

import pandas as pd

KAPPA, ALPHA, BETA = 1.0, 0.3, 0.5
MAX_PER_TOPOLOGY, MAX_PER_METAL, MAX_PER_SUBSTITUENT = 2, 2, 3


def acquisition_score(df: pd.DataFrame, n_dft_per_topology: Dict[str, int],
                      sigma_col: str = "sigma_tree", kappa=KAPPA, alpha=ALPHA, beta=BETA) -> pd.Series:
    novelty = df["topology"].map(lambda t: 1.0 / max(1, n_dft_per_topology.get(t, 0)))
    return (df["mu"] - kappa * df[sigma_col] - alpha * novelty
            + beta * df["is_ood"].astype(float)).rename("acquisition_score")


def select_batch(df: pd.DataFrame, n: int, max_per_topology=MAX_PER_TOPOLOGY,
                 max_per_metal=MAX_PER_METAL, forced: Optional[Iterable[str]] = None,
                 max_per_substituent=MAX_PER_SUBSTITUENT) -> pd.DataFrame:
    """Rank by acquisition score and pick ``n`` candidates under diversity caps.
    ``forced`` names (e.g. closed-shell anchors) are taken first and count toward caps."""
    picked, topo_c, metal_c, sub_c = [], Counter(), Counter(), Counter()
    has_sub = "substituent" in df.columns
    ranked = df.sort_values("acquisition_score")
    for name in (forced or []):
        hits = df.index[df["name"] == name]
        if len(hits):
            idx = hits[0]
            picked.append(idx)
            topo_c[df.at[idx, "topology"]] += 1
            metal_c[df.at[idx, "metal"]] += 1
            if has_sub:
                sub_c[df.at[idx, "substituent"]] += 1
    for idx, r in ranked.iterrows():
        if len(picked) >= n:
            break
        if idx in picked:
            continue
        if topo_c[r.topology] >= max_per_topology or metal_c[r.metal] >= max_per_metal:
            continue
        if has_sub and sub_c[r.substituent] >= max_per_substituent:
            continue
        picked.append(idx)
        topo_c[r.topology] += 1
        metal_c[r.metal] += 1
        if has_sub:
            sub_c[r.substituent] += 1
    out = df.loc[picked].copy()
    out["selection_rank"] = range(1, len(out) + 1)
    return out
