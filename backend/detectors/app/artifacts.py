"""Pinned provenance and file-integrity checks shared by preparation and runtime."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from .errors import ConfigurationError

MODEL_ID = "hivetrace/gliner-guard-omni"
MODEL_REVISION = "bda9b81a77421f48bff373d8339adaf225ff8764"
TOKENIZER_ID = "microsoft/mdeberta-v3-base"
# microsoft's released tokenizer revision (immutable Hub commit).
TOKENIZER_REVISION = "02a99710df3b0711f315e921b990df0191474ef7"
PACKAGES = ("gliner2", "torch", "transformers", "tokenizers", "sentencepiece", "protobuf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    return {name: version(name) for name in PACKAGES}


def verify_manifest(root: Path) -> dict:
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            manifest["model"] != {"id": MODEL_ID, "revision": MODEL_REVISION}
            or manifest["tokenizer"] != {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION}
            or manifest["packages"] != package_versions()
            or not manifest["files"]
        ):
            raise ValueError
        for relative, expected in manifest["files"].items():
            path = root / relative
            if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != expected:
                raise ValueError
        actual = {
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and p.name != "manifest.json" and ".cache" not in p.parts
        }
        if actual != set(manifest["files"]):
            raise ValueError
        return manifest
    except Exception:
        raise ConfigurationError(
            "model manifest: missing, incompatible or corrupt artifacts"
        ) from None
