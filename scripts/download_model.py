"""Download a Hugging Face model (and its tokenizer/custom-code files) into the project's data directory.

Why this script exists
---------------------
The FOK experiment needs per-layer hidden states. The GGUF quantized model available locally cannot
provide those (llama.cpp exposes only logits / final-layer embeddings). We therefore use the original
Hugging Face weights in safetensors form, from which PyTorch/transformers can return hidden states for
every layer via ``output_hidden_states=True``.

The model is downloaded into ``data/models/<repo-name>/`` **inside the project folder**, so the large
weights never end up in an unrelated cache directory and can be easily found.

Usage
-----
    uv run python scripts/download_model.py --repo nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

By default only the files actually needed for inference are fetched (config, weights, tokenizer and
the custom-code modules ``modeling_*.py`` / ``configuration_*.py``). ``--all`` fetches everything.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError, HFValidationError
from huggingface_hub.utils import EntryNotFoundError  # not a top-level export always


def _is_not_found(err: Exception) -> bool:
    """Return True if the exception represents a missing file/repo (404)."""
    return isinstance(err, (EntryNotFoundError, RepositoryNotFoundError, HFValidationError)) or (
        getattr(err, "response", None) is not None
        and getattr(err.response, "status_code", None) == 404
    )

# Project root = two directories up from this script (scripts/ -> project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "Qwen/Qwen3.5-2B"

# Files commonly required to load a model and tokenizer with HF transformers.
# For repos that need trust_remote_code, additional custom-code files are pulled
# automatically by transformers when using --all, or you can add them here.
REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors-00001-of-00001.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model weights into the project data dir.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id to download.")
    parser.add_argument("--revision", default="main", help="Repo revision/branch.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download the whole repo (including README, license, etc.) instead of only required files.",
    )
    args = parser.parse_args()

    # Destination is always inside the project folder: data/models/<repo name>
    local_dir = PROJECT_ROOT / "data" / "models" / args.repo.split("/")[-1]
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fok] Model directory: {local_dir}")

    if args.all:
        snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            local_dir=str(local_dir),
        )
        print("[fok] Snapshot download complete.")
        return

    # Only fetch the files we actually need, one by one. hf_hub_download resolves the
    # xet/LFS redirect for big safetensors files for us. Files that don't exist in a
    # given repo (e.g. no generation_config.json) are skipped with a warning.
    base_url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/"
    import urllib.request

    for fname in REQUIRED_FILES:
        # quick HEAD-style existence check so a 404 does not spam the traceback
        try:
            urllib.request.urlopen(base_url + fname, timeout=30)
        except Exception as e:
            code = getattr(e, "code", None)
            if code == 404:
                print(f"[fok] Skipping missing file: {fname}")
                continue
            # network hiccup - just attempt the real download below anyway
        print(f"[fok] Downloading {fname} ...")
        try:
            hf_hub_download(
                repo_id=args.repo,
                revision=args.revision,
                filename=fname,
                local_dir=str(local_dir),
            )
        except Exception as e:
            if _is_not_found(e):
                print(f"[fok] Skipping missing file: {fname}")
                continue
            raise
        size = os.path.getsize(local_dir / fname) if (local_dir / fname).exists() else 0
        print(f"[fok]   {size / 1e9:.2f} GB")

    print("[fok] Done. Model is ready to be loaded from:")
    print(f"[fok]   {local_dir}")


if __name__ == "__main__":
    main()
