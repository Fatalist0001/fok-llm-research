"""End-to-end integration test of the pipeline on synthetic feature files.

No model is needed: we fabricate a realistic ``features/`` directory exactly as
the ``extract`` stage would produce, then run evaluate -> analyze -> plot and
assert all artifacts appear. This mirrors the real clause but avoids loading
torch/the model in CI.
"""

import csv
import json

import numpy as np

from fok.analysis import analyze
from fok.config import ExperimentConfig
from fok.evaluation import evaluate
from fok.visualization import plot


def _write_features(features):
    features.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    N = 90
    layers = list(range(6))
    hidden = 32
    signal = rng.standard_normal(N)
    rows = []
    for i in range(N):
        split = ["train", "train", "val", "test"][i % 4]
        rows.append({
            "id": f"x{i}", "split": split, "question": f"q{i}",
            "category": "synth", "duration": "0.5",
            "correct_answer": "A", "generated": "A",
            "correct": int((signal[i] + rng.standard_normal() * 0.2) > 0),
            "avg_logprob": round(float(rng.normal(-1, 0.3)), 4),
            "seq_logprob": round(float(rng.normal(-8, 2)), 4),
            "first_entropy": round(float(rng.normal(1, 0.4)), 4),
            "mean_entropy": round(float(rng.normal(2, 0.4)), 4),
            "top1_prob": round(float(abs(rng.normal(0.7, 0.2))), 4),
            "knowable": int(signal[i] > 0),
            "info_relevant": int((signal[i] + rng.standard_normal() * 0.3) > 0),
            "answerable": 1,
        })
    with open(features / "examples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    X = np.zeros((N, len(layers), hidden), dtype=np.float32)
    for li in range(len(layers)):
        X[:, li, :] = (signal[:, None] * np.ones((1, hidden))).astype(np.float32)
        X[:, li, :] += rng.standard_normal((N, hidden)).astype(np.float32) * 0.1
    np.save(features / "hidden_A.npy", X)
    np.save(features / "hidden_B.npy", X)
    with open(features / "extraction_meta.json", "w") as f:
        json.dump({"model_path": "fake", "layers": layers,
                   "representation": "last_token", "n_layers_total": 5,
                   "hidden_dim": hidden, "max_new_tokens": 8, "seed": 1}, f)


def test_end_to_end(tmp_path):
    cfg = ExperimentConfig(name="it", seed=1, dataset="synthetic_knowledge",
                           probe="logistic", layers="all")
    cfg.results_dir = tmp_path
    _write_features(cfg.run_dir() / "features")

    ev = evaluate(cfg)
    assert "info_relevant" in ev["targets"]
    assert "knowable" in ev["targets"]

    arts = analyze(cfg, n_perm=20)
    for name in ("per_layer_auc", "confidence_contrast", "control",
                 "timepoints", "multidim"):
        assert name in arts, f"missing analysis artifact {name}"

    plots = plot(cfg)
    assert "per_layer" in plots
    assert "confidence_contrast" in plots
    assert (tmp_path / cfg.run_id() / "plots" / "confidence_contrast.png").exists()
