"""The only module depending on GLiNER2's pinned processor/output contract."""

import math
from dataclasses import replace
from pathlib import Path

from .artifacts import MODEL_ID, MODEL_REVISION, verify_manifest
from .config import MAX_CANDIDATES, Profile
from .errors import AdapterContractError, ConfigurationError, ServiceError, TraceLimitError
from .observations import ModelMetadata, ModelObservation, ModelRun
from .windowing import plan_windows


def decode_window(result, text, window, profile):
    if not isinstance(result, dict) or set(result) != {"entities"}:
        raise AdapterContractError()
    entities = result["entities"]
    if not isinstance(entities, dict) or any(label not in profile.labels for label in entities):
        raise AdapterContractError()
    observations = []
    for label, values in entities.items():
        if not isinstance(values, list):
            raise AdapterContractError()
        for value in values:
            if (
                not isinstance(value, dict)
                or not {"text", "start", "end", "confidence"} <= value.keys()
            ):
                raise AdapterContractError()
            start, end, score = value["start"], value["end"], value["confidence"]
            if (
                type(start) is not int
                or type(end) is not int
                or type(score) not in (int, float)
                or not math.isfinite(score)
                or not 0 <= score <= 1
                or not 0 <= start < end <= window.end - window.start
                or not isinstance(value["text"], str)
                or text[window.start + start : window.start + end] != value["text"]
            ):
                raise AdapterContractError()
            observations.append(
                ModelObservation(
                    label,
                    profile.public_type(label),
                    window.id,
                    int(window.id[1:]) - 1,
                    window.start + start,
                    window.start + end,
                    float(score),
                )
            )
            if len(observations) > MAX_CANDIDATES:
                raise TraceLimitError("model")
    return observations


class ProcessorBudget:
    def __init__(self, model, schema):
        self.processor = model.processor
        self.schema = schema
        # The library normally appends punctuation; the HTTP contract requires exact input.
        # Instance-scoped override, used equally by preflight and actual extraction.
        self.processor._normalize_text = lambda text: text
        limits = [model.encoder.config.max_position_embeddings]
        token_limit = self.processor.tokenizer.model_max_length
        if isinstance(token_limit, int) and 0 < token_limit < 10**9:
            limits.append(token_limit)
        self.subword_limit = min(limits)
        self.word_limit = model.config.max_len

    def spans(self, text):
        return [(start, end) for _, start, end in self.processor.word_splitter(text)]

    def fits(self, text):
        record = self.processor.transform_record(text, self.schema, max_len=None)
        return len(record.input_ids) <= self.subword_limit and (
            self.word_limit is None or len(record.text_tokens) <= self.word_limit
        )


class GlinerDetector:
    def __init__(self, profile: Profile, model_path: Path):
        # Configuration (including NER_LABELS) was validated before importing/loading torch.
        import torch
        from gliner2 import GLiNER2

        self.torch = torch
        self.profile = profile
        verify_manifest(model_path)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ConfigurationError("CUDA: exactly one visible GPU required")
        self.model = GLiNER2.from_pretrained(str(model_path / "checkpoint"))
        self.model.to(device="cuda:0", dtype=torch.float32)
        self.model.eval()
        if any(
            p.device != torch.device("cuda:0") or p.dtype != torch.float32
            for p in self.model.parameters()
        ):
            raise ConfigurationError("CUDA: parameter placement or dtype")
        self.device = str(next(self.model.parameters()).device)
        self.dtype = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        self.schema = self.model.create_schema().entities(
            entity_types=list(profile.labels),
            threshold=profile.emit_threshold,
        )
        self.budget = ProcessorBudget(self.model, self.schema)
        if not self.budget.fits("a"):
            raise ConfigurationError("NER_LABELS: schema leaves no room for one text token")

    def detect(self, text: str) -> ModelRun:
        blank = not text or text.isspace()
        observations, windows = [], []
        try:
            plan = (
                plan_windows(text, self.budget.spans(text), self.budget.fits) if not blank else ()
            )
            with self.torch.inference_mode():
                for window in plan:
                    part = text[window.start : window.end]
                    result = self.model.extract(
                        part,
                        schema=self.schema,
                        include_spans=True,
                        include_confidence=True,
                    )
                    # Confidence/offset extraction is host-visible; also fence all GPU work.
                    self.torch.cuda.synchronize(0)
                    found = decode_window(result, text, window, self.profile)
                    observations.extend(found)
                    if len(observations) > MAX_CANDIDATES:
                        raise TraceLimitError("model")
                    windows.append(replace(window, candidate_count=len(found)))
        except (ServiceError, ConfigurationError):
            raise
        except Exception as exc:
            # Inspect diagnostics only to classify; never log or return their contents.
            if isinstance(exc, self.torch.cuda.OutOfMemoryError) or any(
                marker in str(exc).lower() for marker in ("cuda", "cublas", "cudnn", "device-side")
            ):
                raise ServiceError("DETECTOR_UNAVAILABLE", 503, "model") from None
            raise ServiceError("INTERNAL_ERROR", 500, "model") from None
        finally:
            # Do not retain request-derived words in the processor's token cache.
            self.model.processor._tokenize_cached.cache_clear()
        return ModelRun(
            tuple(observations),
            ModelMetadata(
                MODEL_ID,
                MODEL_REVISION,
                self.device,
                self.dtype,
                "skipped_blank" if blank else "ok",
                self.profile.emit_threshold,
                self.profile.accept_threshold,
                self.profile.labels,
                tuple(windows),
            ),
        )
