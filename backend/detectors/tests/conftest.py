import pytest

from backend.detectors.app.config import load_profile


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
