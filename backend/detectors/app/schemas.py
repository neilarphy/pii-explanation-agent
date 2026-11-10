from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from .errors import Stage


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DetectRequest(Contract):
    text: StrictStr

    @field_validator("text")
    @classmethod
    def scalar_values(cls, value: str) -> str:
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ValueError("Unicode scalar values required")
        return value


class Entity(Contract):
    id: str
    type: str
    start: int
    end: int
    text: str
    candidate_ids: list[str]


class CandidateBase(Contract):
    id: str
    type: str
    start: int
    end: int
    decision: Literal["accepted", "rejected"]
    entity_id: str | None


class ModelCandidate(CandidateBase):
    source: Literal["model"] = "model"
    raw_label: str
    window_id: str
    score: float = Field(ge=0, le=1)
    reason: Literal["model_threshold_passed", "below_model_threshold"]


class RegexCandidate(CandidateBase):
    source: Literal["regex"] = "regex"
    rule_id: str
    score: None = None
    reason: Literal["rule_passed", "invalid_boundary", "invalid_format", "checksum_failed"]


class WindowTrace(Contract):
    id: str
    start: int
    end: int
    candidate_count: int


class ModelTrace(Contract):
    id: str
    revision: str
    device: str
    dtype: str
    status: Literal["ok", "skipped_blank"]
    emit_threshold: float
    accept_threshold: float
    labels: list[str]
    windows: list[WindowTrace]


class RuleTrace(Contract):
    id: str
    matches: int


class RegexTrace(Contract):
    engine: Literal["python.re"]
    device: Literal["cpu"]
    status: Literal["ok", "skipped_blank"]
    rules: list[RuleTrace]


class Timings(Contract):
    queue: float = Field(ge=0)
    model: float = Field(ge=0)
    regex: float = Field(ge=0)
    merge: float = Field(ge=0)
    total: float = Field(ge=0)


class Trace(Contract):
    profile: str
    policy_version: str
    offset_unit: Literal["unicode_code_point"] = "unicode_code_point"
    text_length: int
    model: ModelTrace
    regex: RegexTrace
    candidates: list[Annotated[ModelCandidate | RegexCandidate, Field(discriminator="source")]]
    timings_ms: Timings


class DetectResponse(Contract):
    schema_version: Literal["1.0"] = "1.0"
    request_id: UUID
    has_pii: bool
    entities: list[Entity]
    trace: Trace


class ErrorDetail(Contract):
    code: str
    failed_stage: Stage
    completed_stages: list[Stage]


class ErrorResponse(Contract):
    schema_version: Literal["1.0"] = "1.0"
    request_id: UUID
    error: ErrorDetail
