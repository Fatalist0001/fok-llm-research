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

    It keeps the fitted model and the layer/representation it was fit on, so the
    evaluation and analysis stages can reference the exact probe configuration.
    """

    def __init__(self, kind: str = "logistic", C: float = 1.0, **kwargs):
        self.kind = kind
        self.C = C
        self.kwargs = kwargs
        if kind == "logistic":
            from sklearn.linear_model import LogisticRegression

            self.clf = LogisticRegression(
                C=C, max_iter=2000, class_weight=None, **kwargs
            )
        elif kind == "ridge":
            from sklearn.linear_model import Ridge

            self.clf = Ridge(**kwargs)
        else:
            raise ValueError(f"unknown probe kind: {kind}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "logistic":
            return self.clf.predict_proba(X)
        y = self.clf.predict(X)
        out = np.zeros((len(X), 2))
        out[:, 0] = 1 - y
        out[:, 1] = y
        return out

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.clf, "decision_function"):
            return self.clf.decision_function(X)
        return self.clf.predict(X)

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
