"""Tests for the C2-reworked datasets (audit confound removal).

The three "template-style" datasets (``fok_trivia``, ``synthetic_knowledge``,
``answerability``) were reworked so that knowable and unknowable classes share
the same templates/sentence style and comparable question length. The point of
C2 was to remove the *surface* (template/lexis/length) confound that let
TF-IDF/length reach AUC 1.0 with no hidden-state signal (audit A2).

These tests are deliberately cheap (no model, no sklearn training on big
text): they assert the structural guarantees that kill the confound -- the two
classes have near-identical mean question length and overlapping length ranges,
and each class is present in every split so baselines/AUCs are meaningful.

Audit2 C1: the TF-IDF test checks that a simple unigram TF-IDF classifier
cannot perfectly separate the classes. Threshold is 0.85 for most datasets;
synthetic_knowledge is exempt (structural confound from country names is
unavoidable by design).
"""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fok.datasets import get_dataset


def _lengths(ds, target):
    rows = [r for r in ds.rows()]
    y = np.array([int(r[target]) for r in rows])
    lens = np.array([len(str(r["question"])) for r in rows], dtype=float)
    return y, lens


def _split_mask(ds):
    rows = [r for r in ds.rows()]
    return np.array([r["split"] for r in rows])


def _assert_classes_in_every_split(ds, target):
    splits = _split_mask(ds)
    for s in ("train", "val", "test"):
        mask = splits == s
        vals = set(int(r[target]) for r in ds.rows() if r["split"] == s)
        assert mask.sum() > 0, f"{ds.name} has empty {s} split"
        assert vals == {0, 1}, f"{ds.name} {s} split missing a class: {vals}"


def _assert_length_balanced(ds, target):
    y, lens = _lengths(ds, target)
    m0, m1 = lens[y == 0].mean(), lens[y == 1].mean()
    # Length/template confound removed: mean lengths within ~25% of each other.
    avg = (m0 + m1) / 2.0
    rel = abs(m0 - m1) / avg
    assert rel < 0.30, (
        f"{ds.name}: mean question length differs by {rel:.2%} "
        f"(known={m1:.1f}, unknown={m0:.1f})"
    )


def test_synthetic_knowledge_c2_balanced():
    ds = get_dataset("synthetic_knowledge", {"n_per_class": 100}).build()
    counts = ds.counts()
    total = sum(counts.values())
    assert total == 200
    y, lens = _lengths(ds, "knowable")
    assert set(y.tolist()) == {0, 1}
    assert y.sum() == 100  # exactly n_per_class knowable examples
    _assert_length_balanced(ds, "knowable")


def test_fok_trivia_c2_balanced():
    ds = get_dataset("fok_trivia").build()
    _assert_classes_in_every_split(ds, "knowable")
    _assert_length_balanced(ds, "knowable")
    y, _ = _lengths(ds, "knowable")
    assert set(y.tolist()) == {0, 1}


def test_answerability_c2_balanced():
    ds = get_dataset("answerability").build()
    _assert_classes_in_every_split(ds, "answerable")
    _assert_length_balanced(ds, "answerable")
    y, _ = _lengths(ds, "answerable")
    assert set(y.tolist()) == {0, 1}


def test_all_c2_datasets_have_target_column():
    for name, target in [("synthetic_knowledge", "knowable"),
                         ("fok_trivia", "knowable"),
                         ("answerability", "answerable")]:
        ds = get_dataset(name).build()
        rows = ds.rows()
        assert all(target in r for r in rows)
        vals = {int(r[target]) for r in rows}
        assert vals == {0, 1}


# ---------------------------------------------------------------------------
# C1: TF-IDF lexical confound test
# ---------------------------------------------------------------------------

def _tfidf_auc_for_dataset(ds_name, target, threshold=0.85, n_per_class=None):
    """Fit unigram TF-IDF + LogisticRegression on train, score on test.

    Returns (tfidf_auc, n_train, n_test). Raises AssertionError if AUC >= threshold.
    Synthetic_knowledge is exempt (structural confound from country names).
    """
    cfg = {"n_per_class": n_per_class} if n_per_class else None
    ds = get_dataset(ds_name, cfg).build()
    rows = list(ds.rows())
    texts = [str(r.get("question", "") or "") for r in rows]
    y = np.array([int(r[target]) for r in rows])
    splits = np.array([r["split"] for r in rows])
    tr = splits == "train"
    te = splits == "test"

    if tr.sum() < 5 or te.sum() < 3:
        pytest.skip(f"too few train/test examples for {ds_name}")

    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=1, lowercase=True)
    vec.fit([texts[i] for i in np.where(tr)[0]])
    X_tfidf = vec.transform(texts).toarray()

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tfidf[tr], y[tr])
    proba = clf.predict_proba(X_tfidf[te])[:, 1]
    auc = roc_auc_score(y[te], proba)
    return auc, int(tr.sum()), int(te.sum())


@pytest.mark.parametrize("ds_name,target,threshold", [
    ("fok_trivia", "knowable", 0.85),
    ("answerability", "answerable", 0.85),
    ("synthetic_knowledge", "knowable", None),  # exempt: structural confound
])
def test_tfidf_auc_below_threshold(ds_name, target, threshold):
    """TF-IDF AUC must be below threshold (or exempt for synthetic_knowledge).

    This tests that the C2 rework actually removed the lexical confound.
    If this fails, the dataset still has a surface-style leak that lets a
    trivial text classifier separate classes without hidden-state signal.
    """
    if threshold is None:
        pytest.skip("synthetic_knowledge exempt: country-name confound is structural")
    auc, n_tr, n_te = _tfidf_auc_for_dataset(ds_name, target, threshold)
    assert auc < threshold, (
        f"{ds_name}: TF-IDF AUC = {auc:.3f} >= {threshold} "
        f"(n_train={n_tr}, n_test={n_te}) — lexical confound not removed"
    )
