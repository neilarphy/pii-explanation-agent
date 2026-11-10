"""Online-only preparation; runtime never downloads or modifies these artifacts."""

import argparse
import json
import os
import shutil
from pathlib import Path

# Compose's offline environment is intentionally overridden ONLY in this preparation process,
# before any HF imports (the libraries cache these settings at import time).
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from app.artifacts import (
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    package_versions,
    sha256,
    verify_manifest,
)


def prepare(output: Path):
    from gliner2.models.base import load_extractor_tokenizer
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    output.mkdir(parents=True, exist_ok=True)
    # A failed preparation never leaves an apparently valid manifest.
    (output / "manifest.json").unlink(missing_ok=True)
    checkpoint = output / "checkpoint"
    base = output / "base_tokenizer"
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=checkpoint,
        allow_patterns=["*.json", "*.safetensors", "encoder_config/*", "*.model", "*.txt"],
    )
    snapshot_download(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
        local_dir=base,
        allow_patterns=[
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "spm.model",
            "added_tokens.json",
        ],
    )
    # The pinned checkpoint ships a fast tokenizer. Preserve its vocabulary and added
    # token IDs; the base SentencePiece asset is a fallback, never a replacement.
    if not (checkpoint / "spm.model").exists():
        shutil.copyfile(base / "spm.model", checkpoint / "spm.model")
    tokenizer = load_extractor_tokenizer(str(checkpoint))
    tokenizer.save_pretrained(checkpoint)
    # Ensure encoder construction is also independent of the user's Hub cache.
    encoder = checkpoint / "encoder_config/config.json"
    if not encoder.is_file():
        AutoConfig.from_pretrained(base, local_files_only=True).save_pretrained(encoder.parent)
    for root in (checkpoint, base):
        shutil.rmtree(root / ".cache", ignore_errors=True)
    files = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "format_version": 1,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION},
        "packages": package_versions(),
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_manifest(output)
    print("Pinned model and tokenizer artifacts prepared and verified.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
