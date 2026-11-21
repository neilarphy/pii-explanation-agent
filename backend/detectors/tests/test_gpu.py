"""Opt-in real model tests: `pytest -m gpu`; no synthetic model scores here."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def detector(request):
    if "gpu" not in request.config.getoption("-m"):
        pytest.skip("select explicitly with -m gpu on the GPU host")
    import torch
    from app.config import load_profile
    from app.gliner_detector import GlinerDetector

    assert torch.cuda.is_available(), "GPU acceptance requires CUDA; no CPU fallback"
    return GlinerDetector(load_profile(), Path(os.environ.get("MODEL_PATH", "/models")))


def test_real_forward_ru_en_and_output_contract(detector):
    texts = [
        "My name is John Smith. Email john.smith@example.org. Call +12025550123.",
        "Меня зовут Анна Иванова. Почта anna.ivanova@example.org. Телефон +79991234567.",
    ]
    observed = 0
    for text in texts:
        run = detector.detect(text)
        assert run.metadata.device == "cuda:0"
        assert run.metadata.dtype == "float32" and run.metadata.status == "ok"
        assert run.metadata.labels == detector.profile.labels
        for obs in run.observations:
            assert 0 <= obs.start < obs.end <= len(text)
            assert 0 <= obs.score <= 1
            assert text[obs.start : obs.end]
        observed += len(run.observations)
    assert observed > 0, "no model spans on the positive RU/EN smoke set"


def test_last_window_really_inferred(detector, monkeypatch):
    text = "neutral word " * 400 + "John Smith: john.smith@example.org"
    called = []
    original = detector.model.extract

    def recording(part, *args, **kwargs):
        called.append(part)
        return original(part, *args, **kwargs)

    monkeypatch.setattr(detector.model, "extract", recording)
    run = detector.detect(text)
    assert len(called) == len(run.metadata.windows) > 1
    last = run.metadata.windows[-1]
    assert last.end == len(text) and called[-1] == text[last.start : last.end]
    for obs in run.observations:
        window = next(w for w in run.metadata.windows if w.id == obs.window_id)
        assert window.start <= obs.start < obs.end <= window.end


@pytest.mark.integration
def test_pinned_real_processor_and_schema(detector):
    text = "🙂 Анна\r\ne\u0301\ue000 a@example.org"
    processor = detector.model.processor
    record = processor.transform_record(text, detector.schema, max_len=None)
    batch = processor.collate_fn_inference([(text, detector.schema)], max_len=None)
    assert record.text == batch.original_texts[0] == text
    assert record.input_ids == batch.input_ids[0, : len(record.input_ids)].tolist()
    assert record.schema["entities"].keys() == dict.fromkeys(detector.profile.labels).keys()
    for start, end in zip(record.start_token_idx, record.end_token_idx, strict=True):
        assert 0 <= start < end <= len(text)
    assert detector.budget.fits(text)
    blank = detector.detect(" \r\n\t")
    assert not blank.observations and not blank.metadata.windows
    assert blank.metadata.labels == detector.profile.labels
