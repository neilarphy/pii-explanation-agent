import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from app.artifacts import (
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    sha256,
    verify_manifest,
)
from app.errors import ConfigurationError


@pytest.mark.parametrize("labels", ["", "person,person", "person,", "first name"])
def test_invalid_labels_exit_before_weights(labels):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "from app.main import initialize; initialize()"],
        cwd=root,
        env={**os.environ, "NER_LABELS": labels},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "NER_LABELS" in result.stderr
    assert "gliner2" not in result.stderr and "torch" not in result.stderr


def test_manifest_integrity_and_package_match(tmp_path, monkeypatch):
    from app import artifacts

    monkeypatch.setattr(artifacts, "package_versions", lambda: {"gliner2": "2.1.0"})
    (tmp_path / "checkpoint").mkdir()
    content = tmp_path / "checkpoint/config.json"
    content.write_text("{}")
    manifest = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION},
        "packages": {"gliner2": "2.1.0"},
        "files": {"checkpoint/config.json": sha256(content)},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert verify_manifest(tmp_path) == manifest
    content.write_text('{"changed":true}')
    with pytest.raises(ConfigurationError):
        verify_manifest(tmp_path)
    content.write_text("{}")
    (tmp_path / "extra.json").write_text("{}")
    with pytest.raises(ConfigurationError):
        verify_manifest(tmp_path)
    (tmp_path / "extra.json").unlink()
    manifest["packages"]["gliner2"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError):
        verify_manifest(tmp_path)


def test_threshold_configuration(profile, tmp_path, monkeypatch):
    from app.config import ROOT, load_profile

    monkeypatch.delenv("NER_LABELS", raising=False)
    source = (ROOT / "config/profile.toml").read_text()
    for changed in ["emit_threshold = 0.0", "emit_threshold = 0.6", "emit_threshold = 1.1"]:
        path = tmp_path / "profile.toml"
        path.write_text(source.replace("emit_threshold = 0.20", changed))
        with pytest.raises(ConfigurationError):
            load_profile(path)
