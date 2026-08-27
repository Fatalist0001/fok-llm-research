# FOK Research

A reproducible research pipeline investigating whether a dense language model's
**internal (per-layer) hidden states** carry a *Feeling of Knowing* (FOK)
signal — information about whether the model "knows" — that is **distinct from**
both token-level confidence and from final correctness.

The pipeline is model- and dataset-agnostic. It currently targets
`Qwen/Qwen3.5-2B` (a dense, non-MoE, native-transformers model that exposes
per-layer hidden states) but works with any dense Hugging Face CausalLM that
supports `output_hidden_states=True`.

---

## The research question

When we ask a model a question, prior to writing any answer, does its internal
state at some layer(s) already "know" whether it can answer? If yes, is that
signal:

1. **per-layer** — localized to particular layers (§4, §12),
2. **distinct from confidence** — not merely a re-encoding of the top-token
   probability / log-probability (§8),
3. **distinct from correctness** — measurable before the answer, and for
   errors and unanswerable questions as well (§11), and
4. **stable across time points** — present before/while/after producing the
   answer (§9)?

We probe these questions with **linear read-outs** (plus a secondary MLP) — a
*diagnostic* that asks "is this information linearly decodable from the hidden
states?", never a claim that linear separability *is* the FOK experience.

## Key design principle: keep the three quantities separate

| Quantity        | What it is                                          | How we measure it                       |
|-----------------|-----------------------------------------------------|-----------------------------------------|
| FOK signal      | internal "knows it can answer" state (the object of research) | linear probe reads hidden states; tested vs. chance (§13) |
| Confidence      | self-reported certainty from the generated text     | token probs, avg/seq log-prob, entropy (baseline, §8)    |
| Correctness     | did the produced answer match a ground-truth answer  | string match against `correct_answer`   |

Confidence and correctness are recorded as **baselines compared to** the hidden
state probe; the probe's *input-condition target* (e.g. `knowable`,
`info_relevant`, `answerable`) is chosen to be **independent of the produced
text**, keeping FOK separable from correctness by construction.

---

## Pipeline

```
extract ──> evaluate ──> analyze ──> plot
```

Each stage is driven by one YAML config (`configs/fok.yaml`) and writes into
`results/<run_id>/`.

| Stage      | Input                     | Outputs                                                        |
|------------|---------------------------|---------------------------------------------------------------|
| `extract`  | model + dataset           | `features/examples.csv`, `features/hidden_{A,B,C}.npy`, meta   |
| `evaluate` | features                  | `eval/probe_results.csv`, `eval/confidence_baselines.csv`      |
| `analyze`  | features + eval           | `analysis/{per_layer_auc,confidence_contrast,control,timepoints,multidim}.csv` |
| `plot`     | analysis                  | `plots/*.png`                                                  |

### extract
Runs the model over every example and, for each, stores:
- **hidden states** at time point **A** (after the question, before
  generation), and optionally **B** (after `b_tokens` answer tokens) and **C**
  (after the full answer) — §9. Only the configured layers and representation
  are kept, per the "don't store activations bigger than needed" rule.
- **the generated answer** and its confidence baselines,
- **correctness** vs. `correct_answer`,
- the **input-condition target labels** from the dataset (e.g. `knowable`,
  `info_relevant`, `answerable`).

### evaluate
Fits one probe **per layer** (on train) and scores it on val/test. Also fits
tiny probes on each **scalar confidence feature** (best-case single scalar
comparison for §8).

### analyze
- `per_layer_auc` — signal as a function of layer (§4, §12).
- `confidence_contrast` — best hidden layer vs best scalar confidence (§8).
- `control` — permutation null: shuffle labels, refit, is the observed AUC
  above the 95th percentile of chance? (§13).
- `timepoints` — AUC at A vs B vs C (§9).
- `multidim` — separability of the FOK target vs. correctness / other
  input-condition targets, as a decoupling check (§11).

### plot
Renders the analysis CSVs to PNG figures.

---

## Usage

```bash
# install (this repo's venv; pick a torch build that fits your GPU)
uv venv .venv
uv pip install -e ".[analysis,test]"

# download the default dense model (already done in this workspace)
fok download-model --repo Qwen/Qwen3.5-2B

# full run with a config
fok run --config configs/fok.yaml

# quick smoke run (small dataset, subset of layers)
fok run --config configs/smoke.yaml

# or drive each stage separately
fok extract  --config configs/fok.yaml
fok evaluate --config configs/fok.yaml
fok analyze  --config configs/fok.yaml --n-perm 200
fok plot     --config configs/fok.yaml
```

Command-line overrides (any value can be overridden without editing the config):

```bash
fok run -c configs/fok.yaml \
    --dataset info_variant \
    --layers all \
    --seed 7
    --dataset-param b_tokens=8
```

Notes:
- `--dataset-param capture_B=true` turns on B/C snapshot capture (needed for
  `analysis/timepoints`); this is slower (extra forward passes).
- `temperature: 0.0` gives deterministic (greedy) generation so answers and
  confidence are comparable across examples.

---

## Datasets

Each dataset exposes one or more binary **input-condition targets** that are
independent of the produced text:

| Dataset              | Target            | Design |
|----------------------|-------------------|--------|
| `fok_trivia`         | `knowable`        | curated pairs of well-known (knowable) and fabricated (unknowable) questions |
| `synthetic_knowledge`| `knowable`        | procedurally scaled: fact banks (knowable) vs invented-entity questions (unknowable) |
| `info_variant`       | `info_relevant`   | the same base question under 4 conditions: no / relevant / irrelevant / misleading info (FOK-mechanism experiment) |
| `answerability`      | `answerable`      | cleanly answerable vs unanswerable questions |

Splits are assigned deterministically by hashing a stable key (the question
text, or the base-question id for `info_variant`) so that near-identical
questions never straddle train and test.

---

## Configuration

The full set of keys (see `src/fok/config.py::ExperimentConfig`):

| Key             | Default | Meaning                                        |
|-----------------|---------|------------------------------------------------|
| `model_path`    | `data/models/Qwen3.5-2B` | local path to the model |
| `layers`        | `all`   | `'all'` or `'0,6,12,18,24'` layer selector      |
| `max_new_tokens`| `96`    | max generated tokens                           |
| `temperature`   | `0.0`   | 0 => greedy                                    |
| `dtype`         | `bfloat16` | model dtype                                  |
| `device`        | `cuda`  | `cuda` or `cpu`                                |
| `dataset`       | `fok_trivia` | dataset name                               |
| `dataset_config`| `{}`    | dataset params (e.g. `n_per_class`)            |
| `representation`| `last_token` | `last_token` / `mean` / `last_k_mean`      |
| `probe`         | `logistic` | `logistic` / `ridge` / `mlp`                 |
| `probe_C`       | `1.0`   | inverse regularization strength               |

---

## Defensibility / controls

- **No train/test leakage**: split keys are hashed, near-identical questions
  stay together and out of overlapping splits.
- **Deterministic generation**: greedy decoding keeps answers comparable.
- **Permutation control** (§13): any reported "signal" is compared to a null
  distribution from shuffled labels.
- **Separate baselines**: confidence and correctness are never conflated with
  the hidden-state probe.

## Project layout

```
configs/      sample YAML configs
data/models/  downloaded models (git-ignored)
results/      per-run artifacts (git-ignored)
scripts/      model downloader
src/fok/
  config.py        ExperimentConfig (single source of truth)
  utils.py         seeds / io helpers
  model/backend.py Hugging Face dense backend (hidden states + confidence)
  datasets/        dataset interface + built-in datasets
  extraction/      collector + answer checking
  probes/          linear / MLP probes
  evaluation/      per-layer metrics + confidence baselines
  analysis/        control, timepoints, multidim, per-layer
  visualization/   plots
  cli.py           click CLI (extract/evaluate/analyze/plot/run)
tests/            pytest suite (pure-python core + synthetic end-to-end)
```
