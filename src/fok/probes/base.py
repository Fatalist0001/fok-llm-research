"""Probe interface and training helpers.

A probe is *not* a definition of FOK; it is a diagnostic instrument. It asks:
"can a simple linear read-out extract information K from these hidden states?"
If yes, the hidden states *contain* information that supports the K-related
distinction. Throughout the code we therefore use neutral names for both the
probe and the signal (``knowledge_state``, ``info_relevant``,
``fok_related_signal``) and avoid claiming that linear separability *is* FOK.

The primary method is a linear probe (logistic regression for binary targets,
ridge regression for continuous targets). A small MLP is offered as a secondary,
more flexible check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class ProbeResult:
    """Everything we keep about a trained probe (for reproducibility)."""

    layer: int
    representation: str
    kind: str                         # "logistic" | "ridge" | "mlp"
    target: Optional[str] = None
    trained_on: str = "train"
    # The fitted sklearn-like estimator
    model: Any = None
    # Fit info
    fit_kwargs: Dict[str, Any] = None
    # Evaluation (filled by the evaluation stage)
    metrics: Dict[str, float] = None
    coef_norm2: float = float("nan")


class LinearProbe:
    """A thin wrapper around sklearn logistic / ridge regression.

    Features are standardised (``StandardScaler``) on the *train* split before
    fitting, and the same transform is applied at inference. Standardising
    removes the dependence of the linear read-out on per-feature scale, which
    matters in the ``p >> n`` regime (small train, very high hidden dimension):
    un-standardised logistic regression is unstable across layers that live on
    different scales.

    Optional C-tuning (``tune_C=True`` + ``X_val``/``y_val`` passed to
    :meth:`fit`) picks ``C`` by validation AUC instead of a fixed value, which
    is the honest treatment of the same ``p >> n`` regime. When tuning is off,
    the fixed ``C`` supplied at construction is used.
    """

    def __init__(
        self,
        kind: str = "logistic",
        C: float = 1.0,
        tune_C: bool = False,
        C_grid: Optional[list] = None,
        **kwargs,
    ):
        self.kind = kind
        self.C = C
        self.tune_C = tune_C
        self.C_grid = C_grid if C_grid is not None else [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
        self.kwargs = kwargs
        self.scaler_ = None
        self.tuned_C_ = None
        self.clf = self._make_clf(C)

    def _make_clf(self, C):
        if self.kind == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(C=C, max_iter=2000, class_weight=None, **self.kwargs)
        elif self.kind == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(**self.kwargs)
        else:
            raise ValueError(f"unknown probe kind: {kind}")

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> "LinearProbe":
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)

        if self.tune_C and X_val is not None and self.kind == "logistic":
            Xs_val = self.scaler_.transform(X_val)
            if len(np.unique(y_val)) >= 2:
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import roc_auc_score

                best_auc, best_c = -1.0, self.C
                for c in self.C_grid:
                    clf = LogisticRegression(C=c, max_iter=2000, class_weight=None, **self.kwargs)
                    clf.fit(Xs, y)
                    try:
                        auc = float(roc_auc_score(y_val, clf.predict_proba(Xs_val)[:, 1]))
                    except Exception:
                        auc = -1.0
                    if auc > best_auc:
                        best_auc, best_c = auc, c
                self.tuned_C_ = best_c
                self.clf = self._make_clf(best_c)
            else:
                self.tuned_C_ = self.C
                self.clf = self._make_clf(self.C)
        elif self.tune_C and self.kind == "logistic":
            self.tuned_C_ = self.C
            self.clf = self._make_clf(self.C)

        self.clf.fit(Xs, y)
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None:
            return X
        return self.scaler_.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self._transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xt = self._transform(X)
        if self.kind == "logistic":
            return self.clf.predict_proba(Xt)
        y = self.clf.predict(Xt)
        out = np.zeros((len(X), 2))
        out[:, 0] = 1 - y
        out[:, 1] = y
        return out

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xt = self._transform(X)
        if hasattr(self.clf, "decision_function"):
            return self.clf.decision_function(Xt)
        return self.clf.predict(Xt)

    def coef(self) -> np.ndarray:
        return np.asarray(self.clf.coef_).reshape(-1)

    def intercept(self) -> float:
        return float(self.clf.intercept_)


class MLPProbe:
    """Small multi-layer perceptron probe (secondary, flexibility check)."""

    def __init__(self, hidden: Tuple[int, ...] = (256,), learning_rate: float = 1e-3, epochs: int = 40, kind: str = "logistic"):
        self.hidden = hidden
        self.lr = learning_rate
        self.epochs = epochs
        self.kind = kind

        import torch
        import torch.nn as nn

        self._torch = torch
        self._nn = nn
        self.model: Any = None

    def _build(self, in_dim: int, out_dim: int):
        nn = self._nn
        layers = []
        prev = in_dim
        for h in self.hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.model = nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPProbe":
        torch = self._torch
        Xt = torch.tensor(X, dtype=torch.float32)
        # regression or classification head
        if self.kind == "logistic":
            yt = torch.tensor(y, dtype=torch.long)
            self.out_dim = 2
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            self.out_dim = 1
            loss_fn = torch.nn.MSELoss()
        self._build(X.shape[1], self.out_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = self.model(Xt)
            if self.kind == "logistic":
                loss = loss_fn(out, yt)
            else:
                loss = loss_fn(out, yt)
            loss.backward()
            opt.step()
        self.model.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32))
            if self.kind == "logistic":
                return torch.softmax(out, dim=-1).numpy()
            p = out.numpy().reshape(-1)
            r = np.zeros((len(p), 2))
            r[:, 1] = np.clip(p, 0, 1)
            r[:, 0] = 1 - r[:, 1]
            return r

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return p[:, 1] > 0.5

    def coef(self) -> np.ndarray:
        return np.asarray(self.model[0].weight.detach().numpy())


def make_probe(kind: str, **kwargs):
    if kind == "mlp":
        return MLPProbe(**kwargs)
    return LinearProbe(kind=kind, **kwargs)


def fit_probe(kind: str, X: np.ndarray, y: np.ndarray, **kwargs):
    """Convenience: build a probe and fit it in one call."""
    return make_probe(kind, **kwargs).fit(X, y)
