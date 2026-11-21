"""Sequential execution and a pure, deterministic decision policy."""

import logging
import math
from dataclasses import asdict
from time import perf_counter
from uuid import UUID

from .config import MAX_CANDIDATES, Profile
from .errors import ServiceError, TraceLimitError
from .observations import (
    ModelDetector,
    ModelObservation,
    ModelRun,
    RegexDetectorInterface,
    RegexRun,
)
from .schemas import (
    DetectResponse,
    Entity,
    ModelCandidate,
    ModelTrace,
    RegexCandidate,
    RegexTrace,
    Timings,
    Trace,
)

logger = logging.getLogger("pii")


def merge_observations(text: str, model: ModelRun, regex: RegexRun, threshold: float):
    """Return entities and full candidates, without clocks, mutation or model calls."""
    observations = [*model.observations, *regex.observations]
    if len(observations) > MAX_CANDIDATES:
        raise TraceLimitError("merge")
    if sum(w.candidate_count for w in model.metadata.windows) != len(model.observations):
        raise ValueError("model candidate count")
    if sum(r.matches for r in regex.metadata.rules) != len(regex.observations):
        raise ValueError("regex candidate count")

    def key(obs):
        is_model = isinstance(obs, ModelObservation)
        return (
            obs.start,
            obs.end,
            obs.type,
            0 if is_model else 1,
            obs.origin_order,
            obs.raw_label if is_model else "",
            -obs.score if is_model else 0,
        )

    candidates = []
    groups = {}
    windows = {w.id: w for w in model.metadata.windows}
    rules = {r.id for r in regex.metadata.rules}
    for index, obs in enumerate(sorted(observations, key=key), 1):
        if not 0 <= obs.start < obs.end <= len(text):
            raise ValueError("observation bounds")
        base = dict(id=f"c{index}", type=obs.type, start=obs.start, end=obs.end, entity_id=None)
        if isinstance(obs, ModelObservation):
            window = windows[obs.window_id]
            if (
                not math.isfinite(obs.score)
                or not 0 <= obs.score <= 1
                or obs.raw_label not in model.metadata.labels
                or not window.start <= obs.start < obs.end <= window.end
            ):
                raise ValueError("model observation")
            accepted = obs.score >= threshold
            candidate = ModelCandidate(
                **base,
                raw_label=obs.raw_label,
                window_id=obs.window_id,
                score=obs.score,
                decision="accepted" if accepted else "rejected",
                reason="model_threshold_passed" if accepted else "below_model_threshold",
            )
        else:
            if obs.rule_id not in rules:
                raise ValueError("regex observation")
            accepted = obs.reason == "rule_passed"
            candidate = RegexCandidate(
                **base,
                rule_id=obs.rule_id,
                decision="accepted" if accepted else "rejected",
                reason=obs.reason,
            )
        candidates.append(candidate)
        if accepted:
            groups.setdefault((obs.start, obs.end, obs.type), []).append(candidate)
    entities = []
    for index, ((start, end, public_type), members) in enumerate(sorted(groups.items()), 1):
        entity_id = f"e{index}"
        for candidate in members:
            candidate.entity_id = entity_id
        entities.append(
            Entity(
                id=entity_id,
                type=public_type,
                start=start,
                end=end,
                text=text[start:end],
                candidate_ids=[candidate.id for candidate in members],
            )
        )
    return entities, candidates


def build_response(text, model, regex, profile, request_id, timings):
    entities, candidates = merge_observations(text, model, regex, profile.accept_threshold)
    return DetectResponse(
        request_id=UUID(str(request_id)),
        has_pii=bool(entities),
        entities=entities,
        trace=Trace(
            profile=profile.name,
            policy_version=profile.policy_version,
            text_length=len(text),
            model=ModelTrace(**asdict(model.metadata)),
            regex=RegexTrace(**asdict(regex.metadata)),
            candidates=candidates,
            timings_ms=timings,
        ),
    )


class Pipeline:
    def __init__(self, model: ModelDetector, regex: RegexDetectorInterface, profile: Profile):
        self.model = model
        self.regex = regex
        self.profile = profile

    def run(self, text: str, request_id: str, started: float, queue_ms: float) -> DetectResponse:
        completed = ["validation", "admission"]
        stage = "model"
        try:
            before = perf_counter()
            model = self.model.detect(text)
            model_ms = (perf_counter() - before) * 1000
            completed.append("model")
            stage = "regex"
            before = perf_counter()
            regex = self.regex.detect(text)
            regex_ms = (perf_counter() - before) * 1000
            if len(model.observations) + len(regex.observations) > MAX_CANDIDATES:
                raise TraceLimitError("regex")
            completed.append("regex")
            stage = "merge"
            before = perf_counter()
            response = build_response(
                text,
                model,
                regex,
                self.profile,
                request_id,
                Timings(
                    queue=queue_ms,
                    model=model_ms,
                    regex=regex_ms,
                    merge=0,
                    total=0,
                ),
            )
            response.trace.timings_ms.merge = (perf_counter() - before) * 1000
            response.trace.timings_ms.total = (perf_counter() - started) * 1000
            return response
        except ServiceError as exc:
            exc.completed = completed.copy()
            raise
        except Exception:
            raise ServiceError("INTERNAL_ERROR", 500, stage, completed) from None
