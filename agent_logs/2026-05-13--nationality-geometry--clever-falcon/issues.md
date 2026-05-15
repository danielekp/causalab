# Issues log — 2026-05-13--nationality-geometry--clever-falcon

## setup-task workflow

### 1. `CausalTrace.intervene()` API in skill template

**Where:** `causalab/causal/trace.py:168` — signature is `intervene(variable: str, value: Any)`, positional args.

**Symptom:** My first counterfactuals.py drafted `.intervene(**{field: new_val})` (kwargs unpack), which failed with `TypeError: CausalTrace.intervene() got an unexpected keyword argument 'country'`.

**Fix applied:** Use positional call `base.copy().intervene(field, new_val)` in
`causalab/tasks/nationality_capitals/counterfactuals.py::_resample_field`.

**Note for the skill:** The setup-task template at `.claude/skills/setup-task/templates/counterfactuals.py` does not demonstrate `.intervene()` usage; the layout doc just says "Use `.copy()` and `.intervene()` on traces" without showing the signature. A worked example in the template would prevent this trap.

### 2. Multi-template token_positions deviates from skill template

**Where:** `.claude/skills/setup-task/instructions/task_package_layout.md` says
`create_token_positions` MUST return `build_token_position_factories()` directly — a `Dict[str, Callable]`. The skill template expects a single-template signature.

**Reality:** For multi-template tasks (template is a causal variable), the loader at
`causalab/tasks/loader.py:90-92` passes `templates=` (list) instead of `template=`,
and the canonical multi-template precedent (`causalab/tasks/natural_domains_arithmetic/token_positions.py`) builds per-template TokenPosition objects with a runtime dispatch on `trace["template"]`, returning a `Dict[str, TokenPosition]` (pre-built, not factories).

**Fix applied:** `causalab/tasks/nationality_capitals/token_positions.py` follows the natural_domains_arithmetic pattern: builds factories per-template via `build_token_position_factories(specs, tmpl)`, calls them once with the pipeline, then wraps them in a dispatcher that selects per-template positions from `input_sample["template"]`.

**Note for the skill:** The task_package_layout doc and the setup-task template should both acknowledge the multi-template branch explicitly. Currently the layout doc reads as universal but the rule only holds for single-template tasks.

### 3. `meta-llama/Llama-3.2-1B-Instruct` access gated; user not authorized

**Symptom:** First validation run failed with `huggingface_hub.errors.GatedRepoError: 403 Client Error … Access to model meta-llama/Llama-3.2-1B-Instruct is restricted`.

**Cache state at time of run:** Llama-3.1-8B was already cached (so user is authorized for that gate), but the 1B-Instruct gate is separate and the user has not been granted access. Other cached models present: Qwen3-0.6B/4B/8B/14B, Qwen2.5-7B-Instruct, OLMo-3-7B, gemma-3-{1b-pt,1b-it,4b-it,12b-it}.

**Fix planned:** Swap the cheap-fallback model. `google/gemma-3-1b-pt` (pretrained, non-gated, ~1B params, already cached) is the natural replacement — pretrained completion model suits our raw next-token task better than an instruction-tuned chat model anyway.

The spec at `causalab/tasks/nationality_capitals/set_up_task.md` still lists Llama-3.2-1B-Instruct in its `models:` block; this should be updated alongside whatever we actually validate against. Cheap-fallback choice is a session-level decision, not a permanent task-spec one.
