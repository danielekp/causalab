"""Focused re-validation: run Llama-3.1-8B on all 40 countries × template 1 only."""

import sys
import time
import json

from causalab.neural.pipeline import LMPipeline
from causalab.tasks.nationality_capitals import (
    causal_model,
    CONTINENT_OF,
    COUNTRIES,
    CAPITAL_OF,
    MAX_TASK_TOKENS,
    MAX_NEW_TOKENS,
    TEMPLATES,
)
from causalab.tasks.nationality_capitals.checker import make_checker

TMPL_IDX = 1
TEMPLATE = TEMPLATES[TMPL_IDX]
print(f"Template 1: {TEMPLATE!r}", flush=True)
print(f"Loading meta-llama/Llama-3.1-8B...", flush=True)
t0 = time.time()
pipeline = LMPipeline(
    "meta-llama/Llama-3.1-8B",
    max_new_tokens=MAX_NEW_TOKENS,
    max_length=MAX_TASK_TOKENS,
    device="cpu",
)
print(f"Loaded in {time.time() - t0:.1f}s", flush=True)
checker = make_checker(pipeline)

correct = 0
misses = []
t_gen0 = time.time()
results = []
for i, country in enumerate(COUNTRIES):
    expected_capital = CAPITAL_OF[country]
    expected = " " + expected_capital
    base = causal_model.sample_input()
    base = base.intervene("country", country).intervene("template", TEMPLATE)
    out = pipeline.generate([base], output_scores=False)
    got = out["string"]
    is_correct = checker(out, expected)
    if is_correct:
        correct += 1
    else:
        misses.append({
            "country": country,
            "continent": CONTINENT_OF[country],
            "expected": expected,
            "got": got[:30],
        })
    results.append({
        "country": country,
        "continent": CONTINENT_OF[country],
        "expected": expected,
        "got": got[:30],
        "correct": is_correct,
    })
    if (i + 1) % 5 == 0:
        elapsed = time.time() - t_gen0
        eta = elapsed / (i + 1) * (len(COUNTRIES) - i - 1)
        print(f"  [{i+1}/{len(COUNTRIES)}] correct={correct} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

acc = correct / len(COUNTRIES)
print(f"\nTemplate 1 accuracy: {correct}/{len(COUNTRIES)} = {acc:.1%}", flush=True)

per_cont: dict[str, list[int]] = {}
for r in results:
    per_cont.setdefault(r["continent"], []).append(int(r["correct"]))
print("\nPer continent:")
for cont, v in sorted(per_cont.items()):
    print(f"  {cont:14s} {sum(v)}/{len(v)} = {sum(v)/len(v):.0%}", flush=True)

if misses:
    print(f"\nMisses ({len(misses)}):")
    for m in misses:
        print(f"  {m['continent']:14s} {m['country']:14s} expected={m['expected']!r:18s} got={m['got']!r}", flush=True)

with open("template1_revalidation.json", "w") as f:
    json.dump({"accuracy": acc, "results": results, "template": TEMPLATE}, f, indent=2)
print("\nSaved template1_revalidation.json", flush=True)
