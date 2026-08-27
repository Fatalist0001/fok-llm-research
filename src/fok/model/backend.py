"""Model backend: wraps a Hugging Face dense LLM and exposes the operations the
research pipeline needs.

The most important contract is that we can obtain **per-layer hidden states**
for a prompt *before* the model has generated an answer (point A), and again
after prefixes of a partially / fully generated answer (points B / C). This is
what makes the FOK investigation possible: we inspect the model's internal
state at a moment when it has "decided" something about its knowledge but has
not yet committed to the final text.

We also expose the usual confidence signals (token probabilities, average and
sequence log-probabilities, entropy) to serve as baselines, and we separate
them from the hidden-state features.

Typical usage
-------------
    backend = HFBackend(cfg)
    snapshot_A = backend.question_snapshot("What is the capital of France?")
    result = backend.generate("What is the capital of France?")
    # result.answer, result.correct, result.confidence, result.hidden_states_B
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import ExperimentConfig
from ..utils import ensure_device, set_seed

logger = logging.getLogger("fok.backend")


@dataclass
class GenerationResult:
    """Output of :meth:`HFBackend.generate` for one question."""

    question: str
    generated_ids: List[int] = field(default_factory=list)
    answer: str = ""
    # Confidence baselines (see README: these are NOT FOK).
    token_probs: List[float] = field(default_factory=list)   # p(top) per token
    avg_logprob: float = float("nan")
    seq_logprob: float = float("nan")
    first_entropy: float = float("nan")
    mean_entropy: float = float("nan")
    # Hidden-state snapshots at the configured time points, keyed by point name.
    # Each snapshot is None unless that point was requested.
    snapshots: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)


@dataclass
class SnapshotSpec:
    """Which time points to capture, and for point B how far into the answer."""

    capture_A: bool = True   # after question, before generation
    capture_B: bool = False  # after `b_tokens` answer tokens
    capture_C: bool = False  # after the full answer
    b_tokens: int = 8


def _softmax_entropy(logits_t: torch.Tensor) -> float:
    """Shannon entropy (in nats) over the softmax of one position's logits."""
    p = torch.softmax(logits_t.float(), dim=-1)
    p = p[p > 0]
    return float(-(p * torch.log(p)).sum().item())


class HFBackend:
    """Holds the tokenizer + model and provides hidden-state / confidence APIs."""

    def __init__(self, config: ExperimentConfig):
        set_seed(config.seed)
        self.config = config
        self.device = ensure_device(config.device)
        logger.info("Loading model from %s (device=%s)", config.model_path, self.device)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        dtype = getattr(torch, config.dtype, torch.bfloat16)
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                dtype=dtype,
                device_map={"": 0},
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path, dtype=dtype, low_cpu_mem_usage=True
            )
        self.model.eval()

        # Layer count used to honor the 'all' selector.
        self.n_layers = self.model.config.num_hidden_layers or 1
        self.hidden_dim = self.model.config.hidden_size
        logger.info(
            "Backend ready: layers=%d hidden_dim=%d params=%.2fB",
            self.n_layers,
            self.hidden_dim,
            sum(p.numel() for p in self.model.parameters()) / 1e9,
        )

    # ------------------------------------------------------------------ #
    # tokenization / prompting
    # ------------------------------------------------------------------ #
    def resolve_layers(self, selector: str) -> List[int]:
        """Turn a layer selector string into concrete layer indices.

        Layer indices refer to *hidden-state* tuples returned by the model with
        ``output_hidden_states=True``: index 0 is the embedding, index 1..N are
        the per-block residual states. The ``-1`` sentinel from 'all' expands to
        the full range.
        """
        from ..config import _resolve_layers

        idx = _resolve_layers(selector)
        if -1 in idx:
            idx = list(range(self.n_layers + 1))  # include embedding slot
        return sorted(set(idx))

    def chat_ids(self, question: str) -> torch.Tensor:
        """Encode a single user question into the model's chat format (no answer)."""
        msgs = [{"role": "user", "content": question}]
        enc = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        enc = enc["input_ids"] if hasattr(enc, "keys") else enc
        return enc.to(self.device)

    # ------------------------------------------------------------------ #
    # hidden-state snapshots
    # ------------------------------------------------------------------ #
    def _represent(self, hidden_states, seq_len: int) -> np.ndarray:
        """Reduce one question's per-layer hidden states to a ``[n_layers, hidden]``
        representation according to ``self.config.representation``.

        ``seq_len`` counts only the *question* (non-padded) tokens when using
        mean pooling, so padding tokens never leak into the representation.
        """
        rep = self.config.representation
        vecs = []
        for hs in hidden_states:
            h = hs[0]                      # [seq, hidden]
            if rep == "last_token":
                v = h[-1]
            elif rep == "mean":
                v = h[:seq_len].mean(dim=0)
            elif rep == "last_k_mean":
                k = min(self.config.last_k, seq_len)
                v = h[seq_len - k:].mean(dim=0)
            else:
                raise ValueError(f"Unknown representation: {rep}")
            vecs.append(v.float().cpu().numpy())
        return np.stack(vecs, axis=0)

    def _question_seq_len(self, question: str) -> int:
        """Token count of the (non-padded) question prefix of the chat prompt.

        The chat prompt is ``[<system/format tags>] user <question> <gen_prompt>``.
        We take the number of tokens up to and including the last question token,
        which is where the "state after reading the question" lives.
        """
        ids = self.chat_ids(question)
        return ids.shape[1]

    def question_snapshot(self, question: str) -> np.ndarray:
        """Hidden states at time point A: after the question, before generation.

        Returns ``[n_layers+1, hidden_dim]`` for the configured representation.
        """
        ids = self.chat_ids(question)
        seq_len = ids.shape[1]
        with torch.no_grad():
            out = self.model(
                input_ids=ids,
                output_hidden_states=True,
                use_cache=False,
            )
        return self._represent(out.hidden_states, seq_len)

    def prefix_snapshot(
        self, prompt_ids: torch.Tensor, answer_ids: torch.Tensor, n_tokens: int
    ) -> np.ndarray:
        """Hidden state of the position just after ``n_tokens`` of the answer have
        been appended to the question. Used for snapshots at points B and C.

        ``seq_len`` here is the full prefix length (question + n answer tokens),
        because the "last token" being probed is the most recent generated token.
        """
        seq_len = prompt_ids.shape[1] + n_tokens
        b_ids = torch.cat(
            [prompt_ids, answer_ids[:, :n_tokens].view(1, -1)], dim=1
        )
        with torch.no_grad():
            out = self.model(input_ids=b_ids, output_hidden_states=True, use_cache=False)
        return self._represent(out.hidden_states, seq_len)

    # ------------------------------------------------------------------ #
    # generation + confidence
    # ------------------------------------------------------------------ #
    def generate(
        self,
        question: str,
        spec: SnapshotSpec | None = None,
    ) -> GenerationResult:
        """Generate an answer and, optionally, capture hidden-state snapshots.

        Generation is greedy (deterministic) by default so that answers and
        confidence measures are comparable across examples. ``output_scores``
        gives us per-step logits from which we compute the confidence baselines.
        """
        from transformers import set_seed as hf_set_seed

        hf_set_seed(self.config.seed)
        ids = self.chat_ids(question)
        spec = spec or SnapshotSpec(capture_A=True)

        gen_kwargs = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, **gen_kwargs)

        gen_ids = out.sequences[0, ids.shape[1]:].tolist()
        answer = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        result = GenerationResult(question=question, generated_ids=gen_ids, answer=answer)

        # Confidence from per-step scores (logits over the vocab).
        logps = []
        probs = []
        entropies = []
        for step_logits in out.scores:
            sl = step_logits[0]  # [vocab]
            p = torch.softmax(sl.float(), dim=-1)
            top_p, top_i = torch.max(p, dim=-1)
            probs.append(float(top_p.item()))
            logps.append(float(torch.log(top_p).item()))
            entropies.append(_softmax_entropy(sl))
        result.token_probs = probs
        result.avg_logprob = float(np.mean(logps)) if logps else float("nan")
        result.seq_logprob = float(np.sum(logps)) if logps else float("nan")
        result.first_entropy = entropies[0] if entropies else float("nan")
        result.mean_entropy = float(np.mean(entropies)) if entropies else float("nan")

        # Hidden-state snapshots at requested time points.
        if spec.capture_A:
            result.snapshots["A"] = self.question_snapshot(question)
        if (spec.capture_B or spec.capture_C) and len(gen_ids) > 0:
            answer_ids = out.sequences[0, ids.shape[1]:].view(1, -1)
            if spec.capture_B:
                n = min(spec.b_tokens, len(gen_ids))
                result.snapshots["B"] = self.prefix_snapshot(ids, answer_ids, n)
            if spec.capture_C:
                result.snapshots["C"] = self.prefix_snapshot(ids, answer_ids, len(gen_ids))
        return result
