"""Model backend for the FOK project.

:mod:`fok.model.backend` wraps a Hugging Face dense LLM and provides per-layer
hidden states at multiple time points plus the confidence baselines.

The backend is model-agnostic (driven by ``config.model_path``); nothing in the
research logic assumes a particular architecture. Dense (non-MoE) transformers
are the intended targets; the hybrid linear/full-attention model used by default
(Qwen3.5-2B) is also handled - it is still a non-MoE model, so no router or
expert activations are involved.
"""
