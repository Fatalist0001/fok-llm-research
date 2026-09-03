# AGENTS.md

Research pipeline for probing "Feeling of Knowing" (FOK) in a dense LLM's per-layer hidden states. Russian-language docs/PLAN; code comments in English. Report summary to user in Russian.

## Commands

- Development CLI: `fok ...` (entry `fok = fok.cli:cli`), also `python -m fok.cli ...`. Install as editable before use.
- Install: `uv pip install -e .` (the venv has **no pip** — it was created with `uv`; never use `pip`).
- Tests: `python -m pytest -q tests/` (lightweight, no model/torch needed). 12 tests, fast.
- End-to-end command order: `fok extract && fok evaluate && fok analyze --n-perm 200 && fok plot`, or `fok run` (does all four).

## Environment gotchas

- `torch` is **NOT** a declared dependency in `pyproject.toml` (GPU-build-specific). It's already installed in `.venv`. Reinstalling via `uv pip install -e .` won't add/alter it.
- Shell is Windows PowerShell. `rg`/`grep` are NOT on PATH — use the Grep/Glob tools, not shell, for searching.
- `.gitignore` excludes: `.venv`, `.hf_cache`, `data/models`, `results`, `PLAN.md`. Model weights (~4.5GB) and run outputs never go in git.
- Model lives at `data/models/Qwen3.5-2B` (native transformers, dense, non-MoE — spec requires dense only). Downloads use `HF_HOME` under `.hf_cache`; no HF token needed.

## Architecture / wiring

- Config is the single source of truth: `src/fok/config.py` `ExperimentConfig`. The YAML must be **flat keys matching the dataclass** (e.g. `model_path`, `dataset`, `layers`, `probe`, `probe_C`). Nested YAML blocks are silently ignored — a common mistake.
- Stage modules: `extraction/collector` (runs model → writes `features/examples.csv` + `hidden_{A,B,C}.npy`), `evaluation/pipeline`, `analysis/pipeline`, `visualization/plots`, wired by `cli.py`.
- Hidden states: index 0 = embedding, 1..N = per-block residual states; `resolve_layers('all')` expands to `n_layers+1`. Points: A = after question/before generation, B/C = after answer tokens (off by default; enable via `dataset_config.capture_B/C=true`).
- **Textual/figure baseline control runs as part of `analyze` (default)** (`analysis/pipeline.py::_simple_baselines`): for each target it fits LogisticRegression on TF-IDF (uni-/bi-grams), question length, and category on the **same** train/val/test splits as the hidden probe. Writes `analysis/simple_baselines.csv` (columns hidden/tfidf/length/category test-AUC). No extra stage or flag needed — it's always produced with `fok analyze` / `fok run`.

## Probe targets (easy to get wrong)

- Probe targets come from **dataset metadata columns**, auto-detected by `evaluation/targets.candidate_targets`: any all-0/1 column present on every row. Actually available: `knowable` (fok_trivia, synthetic_knowledge), `info_relevant` (info_variant), `answerable` (answerability). Probe-all by default; the YAML `target` field is not the operative selector.
- `correct` (answer correctness) is a **reserved** column, not a probe target. It is `NaN`/empty for any question without an expected answer (unknowable/invented) — so `multidim.correct` is NaN on `synthetic_knowledge`/unknowable rows. Use `fok_trivia`/`info_variant` for correctness-relation, not synthetic.

## Known bugs / caveats

- **Layer selection is done on VALIDATION, scored once on test** (`analysis/pipeline.py::_select_layer_by_val`). Never pick a layer by test AUC — that was the old (fixed) test-set leakage. `_control`, `_timepoints`, and `_multidim` all use val-selected layers.
- **Permutation control (`analysis/control`) skips iterations** where shuffling leaves a single class in the test slice (`roc_auc_score` would be NaN). Small datasets yield fewer valid null draws, so their `p_value` is based on fewer permutations.
- `synthetic_knowledge` facts are procedural (fact-bank templates × invented entities); `fok_trivia`/`info_variant`/`answerability` are small hand-curated lists.

## Workflow notes

- Extraction is slow: **~5s/example (A only) to ~10s/example (with B/C)** on the 6GB GPU. Keep smoke runs tiny (`n_per_class=20`, few layers). Full-dataset runs take many minutes.
- Verify preparation work on cheap synthetic data via `tests/test_pipeline.py` (fabricates feature files, runs evaluate→analyze→plot without a model) before spending GPU time.
- Current results and honest status are maintained in `PLAN.md` (git-ignored, user-facing). Read it before claiming what's "done".
