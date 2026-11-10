"""Adapter unit contract with a tiny fake library; never a substitute for -m gpu."""

import re
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from app.errors import AdapterContractError, ConfigurationError, ServiceError
from app.gliner_detector import GlinerDetector


@pytest.fixture
def library(monkeypatch):
    from app import gliner_detector

    state = SimpleNamespace(schema_calls=[], calls=[], mode=False, synced=0, result=None, fail=None)

    class Cache:
        def cache_clear(self):
            state.cleared = True

    class Processor:
        tokenizer = SimpleNamespace(model_max_length=200)
        _tokenize_cached = Cache()

        def word_splitter(self, text):
            return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

        def transform_record(self, text, schema, max_len):
            state.transformed = text
            return SimpleNamespace(
                input_ids=[0] * (len(text) + len(schema.labels) * 2), text_tokens=text.split()
            )

    class Model:
        processor = Processor()
        config = SimpleNamespace(max_len=512)
        encoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=200))

        def parameters(self):
            return iter([SimpleNamespace(device="cuda:0", dtype="float32")])

        def to(self, **kwargs):
            state.placement = kwargs

        def eval(self):
            state.eval = True

        def create_schema(self):
            state.schema_created = getattr(state, "schema_created", 0) + 1
            return self

        def entities(self, *, entity_types, threshold):
            state.schema_calls.append((entity_types, threshold))
            self.labels = entity_types
            return self

        def extract(self, text, *, schema, include_spans, include_confidence):
            assert state.mode and include_spans and include_confidence
            assert schema is self
            state.calls.append((text, threading.get_ident()))
            if state.fail:
                raise state.fail
            return state.result or {"entities": {label: [] for label in self.labels}}

    model = Model()

    class GLiNER2:
        @staticmethod
        def from_pretrained(path):
            state.path = path
            return model

    @contextmanager
    def inference_mode():
        state.mode = True
        try:
            yield
        finally:
            state.mode = False

    class OOM(RuntimeError):
        pass

    def synchronize(device):
        assert device == 0
        state.synced += 1

    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            synchronize=synchronize,
            OutOfMemoryError=OOM,
        ),
        device=lambda value: value,
        float32="float32",
        inference_mode=inference_mode,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "gliner2", SimpleNamespace(GLiNER2=GLiNER2))
    monkeypatch.setattr(gliner_detector, "verify_manifest", lambda path: None)
    state.torch = fake
    return state


@pytest.mark.parametrize("labels", [("person", "passport"), ("employee_id",)])
def test_schema_once_before_inference_and_blank_metadata(profile, library, tmp_path, labels):
    profile = replace(profile, labels=labels)
    detector = GlinerDetector(profile, tmp_path)
    assert library.schema_calls == [(list(labels), 0.2)] and library.schema_created == 1
    assert library.placement == {"device": "cuda:0", "dtype": "float32"} and library.eval
    assert library.path == str(tmp_path / "checkpoint")
    blank = detector.detect(" \t\r\n")
    assert blank.metadata.labels == labels and blank.metadata.status == "skipped_blank"
    assert not library.calls
    run = detector.detect("🙂 exact\r\ne\u0301\ue000 " * 50)
    assert run.metadata.windows[-1].end == len("🙂 exact\r\ne\u0301\ue000 " * 50)
    assert len(library.calls) == library.synced == len(run.metadata.windows) > 1
    assert all(thread == threading.get_ident() for _, thread in library.calls)
    assert len(library.schema_calls) == 1 and library.cleared


def test_labels_exhaust_real_processor_budget(profile, library, tmp_path):
    profile = replace(profile, labels=tuple(f"label_{i}" for i in range(101)))
    with pytest.raises(ConfigurationError, match="NER_LABELS"):
        GlinerDetector(profile, tmp_path)
    assert not library.calls


def test_failure_classification(profile, library, tmp_path):
    detector = GlinerDetector(profile, tmp_path)
    for failure, code in [
        (library.torch.cuda.OutOfMemoryError("private"), "DETECTOR_UNAVAILABLE"),
        (RuntimeError("CUDA device-side failure"), "DETECTOR_UNAVAILABLE"),
        (ValueError("private"), "INTERNAL_ERROR"),
    ]:
        library.fail = failure
        with pytest.raises(ServiceError) as error:
            detector.detect("synthetic")
        assert error.value.code == code and "private" not in str(error.value)


def test_unknown_label_is_explicit_contract_error(profile, library, tmp_path):
    detector = GlinerDetector(profile, tmp_path)
    library.result = {"entities": {"unknown": []}}
    with pytest.raises(AdapterContractError):
        detector.detect("synthetic")


def test_no_gpu_is_startup_failure(profile, library, tmp_path):
    library.torch.cuda.is_available = lambda: False
    with pytest.raises(ConfigurationError, match="CUDA"):
        GlinerDetector(profile, tmp_path)
