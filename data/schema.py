"""Proposed data-evaluation contract; offsets are Python Unicode [start, end)."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Message(Record):
    id: str = Field(min_length=1)
    role: Literal["system", "user", "assistant", "tool"]
    text: str = Field(min_length=1)


class Span(Record):
    id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)


class Detection(Span):
    detector: str = Field(min_length=1)
    detector_version: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    trace: dict[str, Any] = Field(default_factory=dict)


class Verdict(Record):
    label: Literal["TP", "FP", "FN", "TN"]
    detection_id: str | None = None
    gold_entity_id: str | None = None


class Source(Record):
    dataset: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    split: Literal["train", "validation", "calibration", "test"]
    url: str = Field(min_length=1)


class Example(Record):
    schema_version: Literal["1.0"] = "1.0"
    session_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    gold_entities: list[Span]
    # None means not run; [] means a completed run with no detections.
    detections: list[Detection] | None = None
    detector_run_id: str | None = None
    trace_origin: Literal["not_run", "actual", "unit_test"] = "not_run"
    gold_verdicts: list[Verdict] = Field(default_factory=list)
    source: Source
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        def keyed(items: list) -> dict:
            result = {item.id: item for item in items}
            if len(result) != len(items):
                raise ValueError("duplicate IDs")
            return result

        messages = keyed(self.messages)
        gold = keyed(self.gold_entities)
        detections = keyed(self.detections or [])
        for span in [*self.gold_entities, *(self.detections or [])]:
            message = messages.get(span.message_id)
            if message is None or not 0 <= span.start < span.end <= len(message.text):
                raise ValueError("invalid span bounds or message reference")
            if message.text[span.start : span.end] != span.text:
                raise ValueError("span text does not match message")
        if self.detections is None:
            if self.trace_origin != "not_run" or self.detector_run_id or self.gold_verdicts:
                raise ValueError("unexecuted detectors cannot have traces or verdicts")
        elif self.trace_origin == "not_run" or not self.detector_run_id:
            raise ValueError("completed detector runs require provenance and run ID")
        for verdict in self.gold_verdicts:
            d = detections.get(verdict.detection_id)
            g = gold.get(verdict.gold_entity_id)
            if verdict.detection_id is not None and d is None:
                raise ValueError("unknown detection reference")
            if verdict.gold_entity_id is not None and g is None:
                raise ValueError("unknown gold reference")
            expected = {
                "TP": (True, True),
                "FP": (True, False),
                "FN": (False, True),
                "TN": (False, False),
            }
            if (d is not None, g is not None) != expected[verdict.label]:
                raise ValueError("verdict requires different references")
            if verdict.label == "TN" and (gold or detections):
                raise ValueError("TN is session-level and requires no entities or detections")
        return self
