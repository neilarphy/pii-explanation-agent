"""Narrow adapter interface, independent of HTTP, Pydantic and CUDA."""

from dataclasses import dataclass
from typing import Literal, Protocol

Status = Literal["ok", "skipped_blank"]


@dataclass(frozen=True)
class ModelObservation:
    raw_label: str
    type: str
    window_id: str
    origin_order: int
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class RegexObservation:
    rule_id: str
    type: str
    origin_order: int
    start: int
    end: int
    reason: Literal["rule_passed", "invalid_boundary", "invalid_format", "checksum_failed"]


@dataclass(frozen=True)
class Window:
    id: str
    start: int
    end: int
    candidate_count: int = 0


@dataclass(frozen=True)
class ModelMetadata:
    id: str
    revision: str
    device: str
    dtype: str
    status: Status
    emit_threshold: float
    accept_threshold: float
    labels: tuple[str, ...]
    windows: tuple[Window, ...]


@dataclass(frozen=True)
class RuleCount:
    id: str
    matches: int


@dataclass(frozen=True)
class RegexMetadata:
    status: Status
    rules: tuple[RuleCount, ...]
    engine: str = "python.re"
    device: str = "cpu"


@dataclass(frozen=True)
class ModelRun:
    observations: tuple[ModelObservation, ...]
    metadata: ModelMetadata


@dataclass(frozen=True)
class RegexRun:
    observations: tuple[RegexObservation, ...]
    metadata: RegexMetadata


class ModelDetector(Protocol):
    def detect(self, text: str) -> ModelRun: ...


class RegexDetectorInterface(Protocol):
    def detect(self, text: str) -> RegexRun: ...
