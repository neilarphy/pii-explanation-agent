import sys
from pathlib import Path

# Supports both project-root pytest and `cd backend/detectors; pytest`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from app.config import load_profile
from app.observations import ModelMetadata, ModelRun, Window


def pytest_configure(config):
    # Also register when the existing repository-wide pytest config is selected.
    config.addinivalue_line(
        "markers", "gpu: opt-in acceptance with prepared weights and NVIDIA GPU"
    )
    config.addinivalue_line("markers", "integration: pinned real library contract")


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.delenv("NER_LABELS", raising=False)
    return load_profile()


def model_run(profile, observations=(), windows=None, blank=False):
    if windows is None:
        windows = (Window("w1", 0, 100, len(observations)),)
    return ModelRun(
        tuple(observations),
        ModelMetadata(
            "hivetrace/gliner-guard-omni",
            "1" * 40,
            "cuda:0",
            "float32",
            "skipped_blank" if blank else "ok",
            0.2,
            0.5,
            profile.labels,
            tuple(windows),
        ),
    )


class EmptyModel:
    def __init__(self, profile):
        self.profile = profile
        self.calls = []

    def detect(self, text):
        self.calls.append(text)
        blank = not text or text.isspace()
        return model_run(
            self.profile, windows=() if blank else (Window("w1", 0, len(text)),), blank=blank
        )
