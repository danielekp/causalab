#!/usr/bin/env python
"""Phase-1 robustness checks for the country_borders geographic-isomorphism result.

Two post-hoc checks, both operating on the *already-computed* subspace features
(no new model run required):

  1. Statistical null for the (lat, lon) -> PCA combined R^2.
     The headline number (combined R^2 = 1.604) is computed over only ~28
     centroid points, so a permutation null + bootstrap CI are needed before
     the claim is defensible. We permute the country <-> (lat, lon) assignment
     and recompute the best-rotation combined R^2 many times; p-value is the
     fraction of permuted statistics >= observed. Bootstrap resamples countries
     with replacement for a (rough) 95% CI.

  2. East/West residual probe.
     The +0.30 East/West linear-probe lift is argued in REPORT.md to be
     longitude relabeled, not an independent axis. Here we *measure* it:
     linearly regress geography (lat, lon) out of the 32-d centroids and re-run
     the leave-one-out logistic probe on the residuals. If East/West is just
     geography, the residual lift collapses toward ~0. Linguistic family is run
     alongside as a sanity check.

Reproduces the notebook's cell-21 centroid construction exactly so the numbers
are comparable. Writes geometry_robustness.json and geometry_null_hist.png into
the session result/ tree (transported back via the lightweight git path).

Run on the box that has the subspace artifacts (RunPod):

    uv run python agent_logs/2026-05-13--nationality-geometry--clever-falcon/\
code/analyses/geometry_robustness.py

Override paths/iteration counts with --subspace-root / --out / --n-perm /
--n-boot / --seed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score

from causalab.tasks.country_borders import (
    COUNTRIES,
    EAST_WEST_OF,
    LAT_LON_OF,
    LINGUISTIC_FAMILY_OF,
    NEIGHBOR_OF,
)

SESSION_DIR = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Centroid construction — mirrors notebook cell 21 exactly.
# --------------------------------------------------------------------------- #
def build_centroids(subspace_root: Path):
    feat_blob = load_file(subspace_root / "features" / "training_features.safetensors")
    features = feat_blob["features"].cpu().float().numpy()
    with open(subspace_root / "train_dataset.json") as f:
        train_ds = json.load(f)

    example_answer = []
    for ex in train_ds:
        c, d = ex["input"]["country"], ex["input"]["direction"]
        example_answer.append(NEIGHBOR_OF[(c, d)][0] if (c, d) in NEIGHBOR_OF else None)

    centroids = np.zeros((len(COUNTRIES), features.shape[1]))
    n_per = np.zeros(len(COUNTRIES), dtype=int)
    for i, c in enumerate(COUNTRIES):
        rows = [j for j, ans in enumerate(example_answer) if ans == c]
        if rows:
            centroids[i] = features[rows].mean(axis=0)
            n_per[i] = len(rows)

    valid = [i for i, n in enumerate(n_per) if n > 0]
    valid_countries = [COUNTRIES[i] for i in valid]
    centroids_v = centroids[valid]
    n_examples = int(sum(n_per))
    return valid_countries, centroids_v, n_examples


# --------------------------------------------------------------------------- #
# (1) Geographic-isomorphism null.
# --------------------------------------------------------------------------- #
def _combined_r2(P2: np.ndarray, geo: np.ndarray, theta: float) -> float:
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    rot = P2 @ R
    r2x = LinearRegression().fit(geo, rot[:, 0]).score(geo, rot[:, 0])
    r2y = LinearRegression().fit(geo, rot[:, 1]).score(geo, rot[:, 1])
    return r2x + r2y


def best_rotation(P2: np.ndarray, geo: np.ndarray) -> tuple[float, float]:
    """Best in-plane rotation of the 2-D PCA plane onto (lat, lon).

    Returns ``(theta_radians, combined_r2)`` where combined_r2 is
    R^2(geo -> rot_x) + R^2(geo -> rot_y) at the optimum (max 2.0). Public so
    callers that also need the angle (e.g. to draw the rotated map) don't
    re-implement the optimisation.
    """
    res = minimize(lambda t: -_combined_r2(P2, geo, t[0]),
                   x0=[0.0], method="Nelder-Mead")
    return float(res.x[0]), float(-res.fun)


def _best_combined_r2(P2: np.ndarray, geo: np.ndarray) -> float:
    return best_rotation(P2, geo)[1]


def geographic_isomorphism_null(valid_countries, centroids_v,
                                n_perm: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    proj2 = PCA(n_components=2).fit_transform(centroids_v)
    geo = np.column_stack([
        [LAT_LON_OF[c][0] for c in valid_countries],
        [LAT_LON_OF[c][1] for c in valid_countries],
    ])
    n = len(valid_countries)

    t_obs = _best_combined_r2(proj2, geo)

    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _best_combined_r2(proj2, geo[rng.permutation(n)])
    p_val = (1 + int(np.sum(null >= t_obs))) / (1 + n_perm)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        cb = centroids_v[idx]
        # PCA needs >=2 distinct rows; resamples with all-identical rows are
        # vanishingly unlikely at n~28 but guard anyway.
        try:
            pb = PCA(n_components=2).fit_transform(cb)
            boot[i] = _best_combined_r2(pb, geo[idx])
        except Exception:
            boot[i] = np.nan
    boot = boot[~np.isnan(boot)]

    return {
        "n_countries": n,
        "observed_combined_r2": round(t_obs, 4),
        "max_possible": 2.0,
        "permutation": {
            "n_perm": n_perm,
            "p_value": round(p_val, 5),
            "null_mean": round(float(null.mean()), 4),
            "null_p95": round(float(np.percentile(null, 95)), 4),
            "null_max": round(float(null.max()), 4),
        },
        "bootstrap": {
            "n_boot": int(boot.size),
            "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                     round(float(np.percentile(boot, 97.5)), 4)],
            "median": round(float(np.median(boot)), 4),
        },
        "_null_samples": null,  # popped before JSON dump; used for the figure
    }


# --------------------------------------------------------------------------- #
# (2) East/West residual probe.
# --------------------------------------------------------------------------- #
def _loo_lift(X: np.ndarray, y: np.ndarray) -> dict:
    counts = Counter(y)
    mask = np.array([counts[v] >= 2 for v in y])
    Xm, ym = X[mask], y[mask]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    acc = float(cross_val_score(clf, Xm, ym, cv=LeaveOneOut(),
                                scoring="accuracy").mean())
    chance = max(Counter(ym).values()) / len(ym)
    return {"acc": round(acc, 3), "chance": round(chance, 3),
            "lift": round(acc - chance, 3), "n": int(len(ym))}


def residual_probe(valid_countries, centroids_v):
    geo = np.column_stack([
        [LAT_LON_OF[c][0] for c in valid_countries],
        [LAT_LON_OF[c][1] for c in valid_countries],
    ])
    # Linearly remove the (lat, lon) component from every centroid dimension.
    resid = centroids_v - LinearRegression().fit(geo, centroids_v).predict(geo)

    y_ew = np.array([EAST_WEST_OF[c] for c in valid_countries])
    y_fam = np.array([LINGUISTIC_FAMILY_OF[c] for c in valid_countries])
    return {
        "east_west": {"raw": _loo_lift(centroids_v, y_ew),
                      "residual": _loo_lift(resid, y_ew)},
        "linguistic_family": {"raw": _loo_lift(centroids_v, y_fam),
                              "residual": _loo_lift(resid, y_fam)},
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subspace-root", type=Path,
                    default=SESSION_DIR / "artifacts" / "country_borders"
                    / "llama31_8b" / "subspace" / "pca_k32" / "country")
    ap.add_argument("--out", type=Path, default=SESSION_DIR / "result")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (args.subspace_root / "features" / "training_features.safetensors").exists():
        raise SystemExit(
            f"Subspace features not found under {args.subspace_root}. Run this on "
            "the box that holds the subspace artifacts (RunPod), or pass "
            "--subspace-root."
        )

    valid_countries, centroids_v, n_examples = build_centroids(args.subspace_root)
    print(f"centroids: {len(valid_countries)} countries x {centroids_v.shape[1]} "
          f"dims  (from {n_examples} valid examples)\n")

    iso = geographic_isomorphism_null(valid_countries, centroids_v,
                                      args.n_perm, args.n_boot, args.seed)
    null_samples = iso.pop("_null_samples")

    print("=== (1) Geographic-isomorphism null ===")
    print(f"  observed combined R^2 : {iso['observed_combined_r2']} / 2.0")
    print(f"  permutation p-value   : {iso['permutation']['p_value']}  "
          f"(n_perm={iso['permutation']['n_perm']})")
    print(f"  null mean / p95 / max : {iso['permutation']['null_mean']} / "
          f"{iso['permutation']['null_p95']} / {iso['permutation']['null_max']}")
    print(f"  bootstrap 95% CI      : {iso['bootstrap']['ci95']} "
          f"(median {iso['bootstrap']['median']}, n={iso['bootstrap']['n_boot']})\n")

    rp = residual_probe(valid_countries, centroids_v)
    print("=== (2) Residual probe (geography partialled out) ===")
    for axis, d in rp.items():
        print(f"  {axis}")
        print(f"    raw      : lift {d['raw']['lift']:+.3f} "
              f"(acc {d['raw']['acc']}, chance {d['raw']['chance']}, n={d['raw']['n']})")
        print(f"    residual : lift {d['residual']['lift']:+.3f} "
              f"(acc {d['residual']['acc']}, chance {d['residual']['chance']})")
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "geometry_robustness.json").write_text(
        json.dumps({"isomorphism_null": iso, "residual_probe": rp}, indent=2)
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(null_samples, bins=40, color="#7f7f7f", alpha=0.8,
                label=f"permutation null (n={len(null_samples)})")
        ax.axvline(iso["observed_combined_r2"], color="#d62728", lw=2,
                   label=f"observed = {iso['observed_combined_r2']}")
        ax.axvline(iso["permutation"]["null_p95"], color="#1f77b4", ls="--",
                   label=f"null p95 = {iso['permutation']['null_p95']}")
        ax.set_xlabel("best-rotation combined R^2  (lat,lon -> PCA plane)")
        ax.set_ylabel("count")
        ax.set_title("Geographic-isomorphism permutation null "
                     f"(p = {iso['permutation']['p_value']})")
        ax.legend()
        fig.tight_layout()
        (args.out / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out / "figures" / "geometry_null_hist.png", dpi=150)
        print(f"wrote {args.out/'geometry_robustness.json'} and "
              f"figures/geometry_null_hist.png")
    except Exception as e:  # figure is a nice-to-have, JSON is the deliverable
        print(f"(figure skipped: {e})")
        print(f"wrote {args.out/'geometry_robustness.json'}")


if __name__ == "__main__":
    main()
