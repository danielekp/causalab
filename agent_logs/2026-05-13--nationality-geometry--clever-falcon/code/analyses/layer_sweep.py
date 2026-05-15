#!/usr/bin/env python
"""Layer-sweep control for #2 — the decisive experiment (REPORT §10).

§6.4 found, at layer 28 only, a strong Europe map at the relational answer
position but a weak/different one at the direct entity-token position. That
is confounded: the stored representation was never given its best layer.
This reads the per-cell subspace runs produced by layer_sweep_driver.sh and
plots combined (lat,lon)->PCA R^2 vs layer for BOTH read positions.

Decision rule (also in SHARING.md):
  * relational >> direct at EVERY layer  -> the map is constructed by the
    relational computation (strong, novel; the "traversed not stored" claim
    becomes publishable).
  * direct catches up at some layer       -> stored & computed maps live at
    different depths (refined, still interesting, different framing).

Pure post-hoc (CPU). Reuses geometry_robustness (relational answer-country
centroids, null) and retrieval_vs_computation (direct named-country
centroids, geometry helper) so featurisation is identical to §6.3/§6.4.

    uv run python agent_logs/2026-05-13--nationality-geometry--clever-falcon/\
code/analyses/layer_sweep.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SESSION_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SESSION_DIR / "code" / "analyses"))
import geometry_robustness as gr            # noqa: E402
import retrieval_vs_computation as rvc      # noqa: E402

CELL_RE = re.compile(r"^L(\d+)_(last_token|country)$")


def main() -> None:
    ar = SESSION_DIR / "artifacts" / "country_borders" / "llama31_8b"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-root", type=Path, default=ar / "_sweep")
    ap.add_argument("--out", type=Path, default=SESSION_DIR / "result")
    ap.add_argument("--n-perm", type=int, default=500,
                    help="cheaper null per cell; the headline cell can be "
                         "re-confirmed at 2000 separately")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cells = []
    for p in sorted(args.sweep_root.glob("L*_*")):
        m = CELL_RE.match(p.name)
        if not m:
            continue
        root = p / "subspace" / "pca_k32" / "country"
        if (root / "features" / "training_features.safetensors").exists():
            cells.append((int(m.group(1)), m.group(2), root))
    cells.sort(key=lambda t: (t[0], t[1]))
    if not cells:
        raise SystemExit(
            f"No completed sweep cells under {args.sweep_root}. Run "
            "layer_sweep_driver.sh on the GPU box first (see its header)."
        )

    curves: dict[str, dict[int, float]] = {"relational": {}, "direct": {}}
    rows = []
    for layer, pos, root in cells:
        if pos == "last_token":
            cond = "relational"
            countries, C, n = gr.build_centroids(root)          # answer-country
        else:
            cond = "direct"
            countries, C, n = rvc.direct_centroids(root)         # named country
        g = rvc.geometry(countries, C, args.n_perm, args.n_boot, args.seed)
        r2 = g["isomorphism_null"]["observed_combined_r2"]
        p = g["isomorphism_null"]["permutation"]["p_value"]
        curves[cond][layer] = r2
        rows.append((layer, cond, r2, p, g["n"], n))
        print(f"L{layer:<2d} {cond:<10s}  combined R^2 {r2:.3f}  p={p}  "
              f"(n_centroids={g['n']}, n_examples={n})")

    def best(cond):
        d = curves[cond]
        if not d:
            return None
        L = max(d, key=d.get)
        return {"layer": L, "combined_r2": d[L]}

    summary = {
        "relational_curve": curves["relational"],
        "direct_curve": curves["direct"],
        "best_relational": best("relational"),
        "best_direct": best("direct"),
    }
    br, bd = summary["best_relational"], summary["best_direct"]
    print("\n=== best-vs-best ===")
    print(f"  relational best: {br}")
    print(f"  direct best    : {bd}")
    if br and bd:
        verdict = (
            "relational dominates at its best layer too -> 'constructed by "
            "computation' is supported"
            if br["combined_r2"] > bd["combined_r2"] + 0.2 else
            "direct is competitive at some layer -> refine to 'stored & "
            "computed maps at different depths' (do NOT claim traversal)"
        )
        summary["verdict"] = verdict
        print(f"  => {verdict}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "layer_sweep.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for cond, style in (("relational", "o-"), ("direct", "s--")):
            d = curves[cond]
            if d:
                xs = sorted(d)
                ax.plot(xs, [d[x] for x in xs], style, label=cond, lw=2, ms=7)
        ax.axhline(0.55, color="grey", ls=":", lw=1,
                   label="~null max (n=30)")
        ax.set_xlabel("layer")
        ax.set_ylabel("combined R^2  (lat,lon -> PCA plane)  / 2.0")
        ax.set_title("Europe-map strength by layer: relational vs direct read")
        ax.set_ylim(0, 2.0)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        (args.out / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out / "figures" / "layer_sweep.png", dpi=150)
        print(f"\nwrote {args.out/'layer_sweep.json'} and figures/layer_sweep.png")
    except Exception as e:
        print(f"\n(figure skipped: {e}); wrote {args.out/'layer_sweep.json'}")


if __name__ == "__main__":
    main()
