#!/usr/bin/env python
"""Stage B of the steering experiment — causal chord steering (GPU).

Stage A (chord_betweenness.py) showed the Europe manifold is *approximately
locally metric*: for a country B geographically between far A and C, c_B sits
near the chord c_A->c_C (plane r=+0.58, full r=+0.70, p=0.0005; between-vs-off
residuals cleanly separated). That is a *correlational geometric* fact about
the cached representation. Stage B asks the *causal* question the user posed:

  if we linearly interpolate the L28 relational answer-state from a prompt that
  answers A toward one that answers C, does the geographically-between country
  B transiently become the model's answer at the midpoint?

Mechanism (framework-native — causalab's own interpolation intervention):
  new_act = inverse_featurizer( (1-a)*f_base + a*f_src , base_err )
at (layer, last_token) in the SAME PCA-32 subspace as REPORT §6.3 / Stage A.
a=0 is identity, a=1 is full interchange. We sweep a in [0,1] and read the
next-token distribution restricted to country first-tokens
(COUNTRY_FIRST_TOKEN_OF) -> P(A|a), P(B|a), P(C|a).

Prediction if the manifold is causally metric (not just correlationally):
  (1) P(A) decreasing, P(C) increasing, monotone crossover in a;
  (2) P(B) has an INTERIOR peak; argmax_a P(B) ~ t_geo(B) across triples;
  (3) controls kill it: random feature direction, shuffled source country,
      and an off-chord country D (Stage A 'off' set) show no ordered B peak.

Triples are read from result/chord_betweenness.json (the curated
near-collinear, far triples Stage A surfaced) — no hand-coding.

    # 1. wiring check, ZERO GPU (lite pipeline; do this first on RunPod):
    uv run python .../chord_steering.py --dry-run
    # 2. cheap GPU smoke (2 pairs, a in {0,1}); confirms scores format:
    uv run python .../chord_steering.py --smoke
    # 3. full run:
    uv run python .../chord_steering.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

SESSION_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SESSION_DIR / "code" / "analyses"))

from causalab.tasks.country_borders import (  # noqa: E402
    COUNTRIES,
    COUNTRY_FIRST_TOKEN_OF,
    NEIGHBOR_OF,
)
from causalab.tasks.country_borders.causal_models import causal_model  # noqa: E402
from causalab.tasks.country_borders.templates import TEMPLATES  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B"
TARGET_VARIABLE = "country"          # task default; residual target is set by grid
POS = "last_token"                   # the relational answer position (§6.3)


# --------------------------------------------------------------------------- #
# Prompt pools: bucket valid (country, direction, template) inputs by the
# answer country (NEIGHBOR_OF[(country,direction)][0]) — identical labelling
# to Stage A / §6.3 build_centroids.
# --------------------------------------------------------------------------- #
def answer_buckets() -> dict[str, list]:
    """answer-country -> list of CausalTrace inputs. Built via the task's own
    sample_input() scaffold + .intervene() (the pattern counterfactuals.py
    uses) so `template` is the real template *string* and the framework can
    tokenize / build token positions from each input."""
    scaffold = causal_model.sample_input()
    buckets: dict[str, list] = {c: [] for c in COUNTRIES}
    for (country, direction), neigh in NEIGHBOR_OF.items():
        ans = neigh[0]
        if ans not in buckets:
            continue
        for tmpl in TEMPLATES:
            inp = (scaffold.copy()
                   .intervene("country", country)
                   .intervene("direction", direction)
                   .intervene("template", tmpl))
            buckets[ans].append(inp)
    return buckets


def load_triples(path: Path, top_k: int) -> list[dict]:
    blob = json.loads(path.read_text())
    tri = blob.get("illustrative_triples", [])
    if not tri:
        raise SystemExit(
            f"no illustrative_triples in {path}; run Stage A first."
        )
    # interior t_geo first (clean midpoint demo), then most collinear
    tri = sorted(tri, key=lambda r: (abs(r["t_geo"] - 0.5),
                                     -r.get("collinearity", 0.0)))
    return tri[:top_k]


def country_token_ids(tokenizer) -> dict[str, int]:
    """country -> first-BPE-token vocab id of its answer string (e.g.
    ' France'). Matches the task's first_token_only readout semantics; 29 are
    single-token, Czech Republic's first sub-token is ' Czech'."""
    ids = {}
    for c in COUNTRIES:
        s = COUNTRY_FIRST_TOKEN_OF.get(c)
        if not s:
            continue
        enc = tokenizer.encode(s, add_special_tokens=False)
        if enc:
            ids[c] = int(enc[0])
    return ids


# --------------------------------------------------------------------------- #
def _first_step_logits(scores) -> np.ndarray:
    """Normalise run_interpolation_interventions' 'scores' (list of per-batch
    outputs, full vocab on CPU) to a single [N, vocab] array of the FIRST
    decoded token's logits. Defensive about nesting/torch-vs-numpy because the
    exact shape is only knowable at runtime — the --smoke path validates it
    cheaply before the full sweep."""
    import torch

    def to_np(x):
        return x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) \
            else np.asarray(x, dtype=np.float32)

    rows = []
    for batch in scores:
        b = batch
        # generation scores commonly arrive as (steps, [B, vocab]) or [B, vocab]
        while isinstance(b, (list, tuple)):
            b = b[0]
        arr = to_np(b)
        if arr.ndim == 1:
            arr = arr[None, :]
        rows.append(arr)
    return np.concatenate(rows, axis=0)


def country_probs(scores, tok_ids: dict[str, int]) -> dict[str, np.ndarray]:
    """Per-example P(country) = softmax over the country first-token logits."""
    logits = _first_step_logits(scores)                 # [N, vocab]
    order = list(tok_ids)
    sub = logits[:, [tok_ids[c] for c in order]]        # [N, n_country]
    sub = sub - sub.max(axis=1, keepdims=True)
    p = np.exp(sub)
    p /= p.sum(axis=1, keepdims=True)
    return {c: p[:, i] for i, c in enumerate(order)}


# --------------------------------------------------------------------------- #
def main() -> None:
    ar = SESSION_DIR / "artifacts" / "country_borders" / "llama31_8b"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subspace-root", type=Path,
                    default=ar / "subspace" / "pca_k32" / "country")
    ap.add_argument("--triples-json", type=Path,
                    default=SESSION_DIR / "result" / "chord_betweenness.json")
    ap.add_argument("--out", type=Path, default=SESSION_DIR / "result")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--k-features", type=int, default=32)
    ap.add_argument("--n-triples", type=int, default=5)
    ap.add_argument("--max-pairs", type=int, default=24,
                    help="base(A)×source(C) prompt pairs per triple")
    ap.add_argument("--alphas", type=str, default="0,0.2,0.4,0.5,0.6,0.8,1.0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--dry-run", action="store_true",
                    help="lite pipeline (no weights, ZERO GPU): validate task/"
                         "targets/featurizer/dataset/readout wiring, then exit")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny GPU run (2 pairs/triple, alphas {0,1}) to "
                         "confirm the scores format before the full sweep")
    ap.add_argument("--featurizer", choices=["subspace", "identity"],
                    default="subspace",
                    help="'subspace' = interpolate only the §6.3 PCA-32 slice; "
                         "'identity' = interpolate the FULL residual at "
                         "(layer,last_token) — the strongest interchange, used "
                         "as the A->C efficacy anchor")
    ap.add_argument("--anchor", action="store_true",
                    help="endpoint-only efficacy test: alphas {0,1} (does a "
                         "full interchange even flip A->C?). Pair with "
                         "--featurizer identity")
    args = ap.parse_args()

    alphas = ([0.0, 1.0] if (args.smoke or args.anchor)
              else [float(x) for x in args.alphas.split(",")])
    max_pairs = 2 if args.smoke else args.max_pairs
    rng = np.random.default_rng(args.seed)

    triples = load_triples(args.triples_json, args.n_triples)
    buckets = answer_buckets()

    from causalab.runner.helpers import resolve_task, build_targets_for_grid
    task, _ = resolve_task("country_borders", {}, TARGET_VARIABLE, args.seed)

    # ----- pipeline (lite for --dry-run, real otherwise) ------------------- #
    if args.dry_run:
        from causalab.io.pipelines import load_lite_pipeline
        pipeline = load_lite_pipeline(args.model, max_new_tokens=1)
    else:
        from causalab.io.pipelines import load_pipeline
        pipeline = load_pipeline(args.model, task, max_new_tokens=1)

    tok_ids = country_token_ids(pipeline.tokenizer)

    targets, tp_list = build_targets_for_grid(
        pipeline, task, [args.layer], [POS])
    interchange_target = next(iter(targets.values()))
    token_pos = tp_list[0]

    if args.featurizer == "subspace":
        from causalab.analyses.subspace.loading import load_subspace_onto_target
        load_subspace_onto_target(
            interchange_target, str(args.subspace_root), "pca",
            args.k_features)
        kdesc = f"k={args.k_features}"
    else:
        # leave the unit's default identity featurizer -> interpolate the
        # FULL residual (no low-energy 32-d bottleneck). alpha=1 transplants
        # the entire source last_token activation: the strongest possible
        # A->C interchange, used as the efficacy anchor.
        kdesc = "full-residual"
    feat = interchange_target.flatten()[0].featurizer
    print(f"featurizer: {type(feat).__name__}  layer={args.layer} "
          f"pos={token_pos.id}  {kdesc}")

    # ----- build per-triple counterfactual datasets ----------------------- #
    def pairs_for(a_country: str, c_country: str) -> list[dict]:
        ba = list(buckets.get(a_country, []))
        bc = list(buckets.get(c_country, []))
        if not ba or not bc:
            return []
        rng.shuffle(ba)
        rng.shuffle(bc)
        n = min(len(ba), len(bc), max_pairs)
        return [{"input": ba[i], "counterfactual_inputs": [bc[i]]}
                for i in range(n)]

    plan = []
    for r in triples:
        A, B, C = r["A"], r["B"], r["C"]
        ds = pairs_for(A, C)
        plan.append({"A": A, "B": B, "C": C, "t_geo": r["t_geo"],
                     "n_pairs": len(ds), "ds": ds})
        print(f"  {A:>12s} -> {B:>12s} -> {C:<12s} "
              f"t_geo={r['t_geo']:.2f}  pairs={len(ds)}")

    if args.dry_run:
        ok = all(p["n_pairs"] > 0 for p in plan) and len(tok_ids) >= 25
        print(f"\n[dry-run] task/targets/featurizer/dataset/readout wired. "
              f"country readout tokens={len(tok_ids)}  "
              f"triples_ok={'YES' if ok else 'NO'}")
        print("[dry-run] no GPU used. If YES, run --smoke next.")
        return

    # ----- the interpolation sweep ---------------------------------------- #
    from causalab.neural.activations.interpolate import (
        run_interpolation_interventions,
    )

    def linear_interp(f_base, f_src, alpha):
        return (1.0 - alpha) * f_base + alpha * f_src

    results = []
    for p in plan:
        if not p["ds"]:
            continue
        A, B, C = p["A"], p["B"], p["C"]
        curve = {"A": A, "B": B, "C": C, "t_geo": p["t_geo"],
                 "n_pairs": p["n_pairs"], "alphas": alphas,
                 "P_A": [], "P_B": [], "P_C": []}
        for a in alphas:
            out = run_interpolation_interventions(
                pipeline, p["ds"], interchange_target,
                fn=linear_interp, params={"alpha": a},
                batch_size=args.batch_size, output_scores=True,
            )
            pr = country_probs(out["scores"], tok_ids)
            curve["P_A"].append(float(np.mean(pr[A])) if A in pr else None)
            curve["P_B"].append(float(np.mean(pr[B])) if B in pr else None)
            curve["P_C"].append(float(np.mean(pr[C])) if C in pr else None)
        pb = np.array(curve["P_B"], float)
        interior = pb[1:-1]
        a_star = alphas[1 + int(np.argmax(interior))] if interior.size else None
        curve["argmax_alpha_B"] = a_star
        curve["B_interior_peak"] = bool(
            interior.size and interior.max() >= pb[0] and interior.max() >= pb[-1]
        )
        results.append(curve)
        print(f"[{A}->{B}->{C}] t_geo={p['t_geo']:.2f} "
              f"argmax_a P(B)={a_star}  interior_peak={curve['B_interior_peak']} "
              f"P_B={['%.2f' % x for x in curve['P_B']]}")
        if args.smoke:
            print("[smoke] scores format OK — safe to run the full sweep.")
            break

    # cross-triple test: does the B-peak location track geography?
    pts = [(c["t_geo"], c["argmax_alpha_B"]) for c in results
           if c["argmax_alpha_B"] is not None]
    summary = {"layer": args.layer, "model": args.model, "alphas": alphas,
               "triples": results}
    if len(pts) >= 3:
        tg, aa = np.array([x[0] for x in pts]), np.array([x[1] for x in pts])
        summary["corr_tgeo_argmaxB"] = round(
            float(np.corrcoef(tg, aa)[0, 1]), 3)
        summary["n_interior_peaks"] = int(
            sum(c["B_interior_peak"] for c in results))
        print(f"\ncorr(t_geo, argmax_a P(B)) = "
              f"{summary['corr_tgeo_argmaxB']}  "
              f"interior peaks {summary['n_interior_peaks']}/{len(results)}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chord_steering.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(results)
        fig, axes = plt.subplots(1, max(n, 1), figsize=(4.2 * max(n, 1), 4),
                                 squeeze=False)
        for ax, c in zip(axes[0], results):
            ax.plot(alphas, c["P_A"], "o-", label=f"P({c['A']})", color="tab:red")
            ax.plot(alphas, c["P_B"], "s-", label=f"P({c['B']})",
                    color="tab:green", lw=2)
            ax.plot(alphas, c["P_C"], "^-", label=f"P({c['C']})",
                    color="tab:blue")
            ax.set(xlabel="alpha (chord A->C)", ylabel="P(country)",
                   title=f"{c['A']}->{c['B']}->{c['C']} (t_geo={c['t_geo']:.2f})")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        (args.out / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out / "figures" / "chord_steering.png", dpi=150)
        print(f"\nwrote {args.out/'chord_steering.json'} and "
              f"figures/chord_steering.png")
    except Exception as e:
        print(f"\n(figure skipped: {e}); wrote {args.out/'chord_steering.json'}")


if __name__ == "__main__":
    main()
