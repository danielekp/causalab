"""Model validation for the nationality_capitals task.

Loads a single model, runs N prompts, and reports:
- Overall first-token accuracy
- Per-template accuracy
- Per-continent accuracy
- Token-alignment sanity check for 3 examples

Usage:
    python agent_logs/.../run/validate_task.py <model_name> [n_examples]
"""

from __future__ import annotations

import json
import sys
import time
import random
from collections import defaultdict

from causalab.neural.pipeline import LMPipeline
from causalab.tasks.nationality_capitals import (
    causal_model,
    CONTINENT_OF,
    MAX_TASK_TOKENS,
    MAX_NEW_TOKENS,
    TEMPLATES,
)
from causalab.tasks.nationality_capitals.checker import make_checker


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.2-1B-Instruct"
    n_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    random.seed(seed)

    print(f"=== Validating {model_name} on {n_examples} examples ===")
    print(f"Loading model... (this is the slow step on CPU)")
    t0 = time.time()
    pipeline = LMPipeline(
        model_name,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=MAX_TASK_TOKENS,
        device="cpu",
    )
    load_s = time.time() - t0
    print(f"Loaded in {load_s:.1f}s. Device: {pipeline.model.device}")

    checker = make_checker(pipeline)

    correct = 0
    per_template: dict[int, list[int]] = defaultdict(list)  # tmpl_idx -> [0,1,1,...]
    per_continent: dict[str, list[int]] = defaultdict(list)
    misses: list[dict] = []
    token_align: list[dict] = []

    print(f"\nRunning {n_examples} examples...")
    t_gen0 = time.time()
    for i in range(n_examples):
        s = causal_model.sample_input()
        country = s["country"]
        template = s["template"]
        expected = s["raw_output"]
        tmpl_idx = TEMPLATES.index(template)
        cont = CONTINENT_OF.get(country, "?")

        out = pipeline.generate([s], output_scores=False)
        is_correct = checker(out, expected)

        if is_correct:
            correct += 1
        else:
            if len(misses) < 12:
                misses.append({
                    "country": country,
                    "tmpl": tmpl_idx,
                    "expected": expected,
                    "got": out["string"][:40],
                })

        per_template[tmpl_idx].append(int(is_correct))
        per_continent[cont].append(int(is_correct))

        if i < 3:
            actual_ids = pipeline.tokenizer.encode(out["string"], add_special_tokens=False)
            expected_ids = pipeline.tokenizer.encode(expected, add_special_tokens=False)
            token_align.append({
                "input": s["raw_input"],
                "expected": expected,
                "got": out["string"],
                "expected_ids": expected_ids[:5],
                "actual_ids": actual_ids[:5],
                "first_token_match": actual_ids[:1] == expected_ids[:1],
            })

        if (i + 1) % 8 == 0:
            elapsed = time.time() - t_gen0
            eta = elapsed / (i + 1) * (n_examples - i - 1)
            print(f"  [{i+1}/{n_examples}] correct={correct} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    total_s = time.time() - t_gen0
    acc = correct / n_examples
    print(f"\n=== Results ===")
    print(f"Overall: {correct}/{n_examples} = {acc:.1%} (gen time {total_s:.0f}s, {total_s/n_examples:.1f}s/prompt)")

    print(f"\nPer template:")
    for idx in sorted(per_template):
        v = per_template[idx]
        print(f"  [{idx}] {sum(v)}/{len(v)} = {sum(v)/len(v):.0%} | {TEMPLATES[idx][:60]!r}")

    print(f"\nPer continent:")
    for cont, v in sorted(per_continent.items()):
        print(f"  {cont:14s} {sum(v)}/{len(v)} = {sum(v)/len(v):.0%}")

    print(f"\nToken alignment (first 3):")
    for a in token_align:
        match = "OK " if a["first_token_match"] else "MISS"
        print(f"  [{match}] {a['input']!r} -> got={a['got']!r} expected={a['expected']!r}")
        print(f"          expected_ids={a['expected_ids']} actual_ids={a['actual_ids']}")

    if misses:
        print(f"\nFirst {len(misses)} misses:")
        for m in misses:
            print(f"  tmpl{m['tmpl']} {m['country']:14s} expected={m['expected']!r:18s} got={m['got']!r}")

    out_path = sys.argv[1].replace("/", "_").replace("-", "_") + "_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name,
            "n": n_examples,
            "correct": correct,
            "accuracy": acc,
            "load_s": load_s,
            "gen_s": total_s,
            "per_template": {k: sum(v) / len(v) for k, v in per_template.items()},
            "per_continent": {k: sum(v) / len(v) for k, v in per_continent.items()},
            "token_align": token_align,
            "misses": misses,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
