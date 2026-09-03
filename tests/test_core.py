"""Tests for the pure-python core: config, datasets, targets, probes, metrics."""

import numpy as np
import pytest

from fok.config import ExperimentConfig, _resolve_layers, load_config
from fok.datasets import get_dataset
from fok.evaluation.metrics import binary_metrics
from fok.evaluation.targets import candidate_targets
from fok.extraction.answers import check_answer
from fok.probes.base import make_probe


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_resolve_layers():
    assert _resolve_layers("all") == [-1]
    assert _resolve_layers("0,2,4") == [0, 2, 4]
    assert _resolve_layers("0-3") == [0, 1, 2, 3]


def test_config_roundtrip(tmp_path):
    cfg = ExperimentConfig(dataset="synthetic_knowledge", seed=3)
    cfg.dataset_config["n_per_class"] = 20
    p = tmp_path / "cfg.yaml"
    cfg.save(p)
    cfg2 = load_config(p)
    assert cfg2.seed == 3
    assert cfg2.dataset == "synthetic_knowledge"
    assert cfg2.dataset_config["n_per_class"] == 20
    assert cfg2.run_id() == cfg.run_id()


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def test_dataset_splits_and_targets():
    ds = get_dataset("synthetic_knowledge", {"n_per_class": 20})
    counts = ds.counts()
    assert set(counts) == {"train", "val", "test"}
    assert all(c >= 0 for c in counts.values())
    ids = [e.id for e in ds.examples]
    assert len(ids) == len(set(ids))  # unique ids


@pytest.mark.parametrize("name,target_col", [
    ("fok_trivia", "knowable"),
    ("synthetic_knowledge", "knowable"),
    ("info_variant", "info_relevant"),
    ("answerability", "answerable"),
    ("graded_trivia", "knowable"),
])
def test_target_detection(name, target_col):
    ds = get_dataset(name, {"n_per_class": 8} if name == "synthetic_knowledge" else None)
    rows = ds.rows()
    targets = candidate_targets(rows)
    assert target_col in targets


def test_no_train_test_leakage():
    ds = get_dataset("info_variant")
    tr = {e.id for e in ds.examples if e.split == "train"}
    te = {e.id for e in ds.examples if e.split == "test"}
    # same base question must not land in both train and test
    tr_base = {e.metadata["base_id"] for e in ds.examples if e.split == "train"}
    te_base = {e.metadata["base_id"] for e in ds.examples if e.split == "test"}
    assert tr_base.isdisjoint(te_base)


# --------------------------------------------------------------------------- #
# answers
# --------------------------------------------------------------------------- #
def test_check_answer():
    assert check_answer("The capital is Paris.", "Paris") is True
    assert check_answer("red", "blue") is False
    assert check_answer("anything", None) is None
    assert check_answer("", "Paris") is None
    # C1: article stripping + word-boundary matching
    assert check_answer("The capital is Paris.", "the paris") is True
    assert check_answer("It is Paris city.", "paris") is True
    assert check_answer("A person from Parisian descent", "Paris") is False
    assert check_answer("The answer is New York.", "New York") is True


# --------------------------------------------------------------------------- #
# probes + metrics
# --------------------------------------------------------------------------- #
def test_probe_and_metrics():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 32))
    y = (X[:, 0] + rng.standard_normal(200) * 0.4 > 0).astype(int)
    pr = make_probe("logistic")
    pr.fit(X[:100], y[:100])
    p = pr.predict_proba(X[100:])[:, 1]
    m = binary_metrics(y[100:], p)
    assert m["auc"] > 0.6
    assert 0.0 <= m["acc"] <= 1.0


def test_mlp_probe_runs():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 12))
    y = (X[:, 0] > 0).astype(int)
    pr = make_probe("mlp", hidden=(16,), epochs=5)
    pr.fit(X[:40], y[:40])
    # just must not raise; small data / few epochs
    out = pr.predict_proba(X[40:])
    assert out.shape[1] == 2


# --------------------------------------------------------------------------- #
# B1: standardisation, B2: C-tuning (audit fixes)
# --------------------------------------------------------------------------- #
def test_probe_standardises_features():
    """B1 fix: LinearProbe should standardise features on a very skewed scale."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 16)) * rng.exponential(100, (16,))
    y = (X[:, 0] > np.median(X[:, 0])).astype(int)
    pr = make_probe("logistic", C=1.0)
    pr.fit(X[:60], y[:60], X_val=X[60:], y_val=y[60:])
    assert pr.scaler_ is not None
    # prediction uses the scaler, so it must not crash even with huge scales
    p = pr.predict_proba(X[60:])[:, 1]
    assert np.all(np.isfinite(p))


def test_probe_c_tuning_uses_validation():
    """B2 fix: with tune_C=True the probe picks C by validation AUC."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((90, 20))
    # strong but high-dimensional-ish separable signal
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(90) * 0.1 > 0).astype(int)
    pr = make_probe("logistic", C=1.0, tune_C=True)
    pr.fit(X[:60], y[:60], X_val=X[60:75], y_val=y[60:75])
    assert pr.tuned_C_ is not None
    assert pr.tuned_C_ in pr.C_grid
    p = pr.predict_proba(X[75:])[:, 1]
    # probe should still work after tuning
    assert np.all(np.isfinite(p))


def test_probe_c_tuning_off_default():
    """Without tune_C, C stays at its fixed value (no validation needed)."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((60, 12))
    y = (X[:, 0] > 0).astype(int)
    pr = make_probe("logistic", C=1.0)
    pr.fit(X[:40], y[:40])  # no X_val passed
    assert pr.tuned_C_ is None
    assert pr.clf.C == 1.0
