import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]
MAX_CANDIDATES = 4096
MAX_WINDOWS = 512
MAX_TEXT = 32768
MAX_BODY = 512 * 1024


def parse_labels(value: str | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    labels = defaults if value is None else tuple(part.strip() for part in value.split(","))
    if (
        not labels
        or any(re.fullmatch("[a-z][a-z0-9_]*", label) is None for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ConfigurationError("NER_LABELS: invalid ordered label list")
    return labels


@dataclass(frozen=True)
class Rule:
    id: str
    type: str
    pattern: str
    validator: str


@dataclass(frozen=True)
class Profile:
    name: str
    policy_version: str
    emit_threshold: float
    accept_threshold: float
    labels: tuple[str, ...]
    public_types: dict[str, str]
    rules: tuple[Rule, ...]

    def public_type(self, label: str) -> str:
        return self.public_types.get(label, label.upper())


def load_profile(path: Path = ROOT / "config/profile.toml") -> Profile:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    emit, accept = data["emit_threshold"], data["accept_threshold"]
    if not 0 < emit <= accept <= 1:
        raise ConfigurationError("profile: invalid thresholds")
    labels = parse_labels(os.environ.get("NER_LABELS"), tuple(data["default_ner_labels"]))
    rules = tuple(Rule(**rule) for rule in data["rules"])
    if len({rule.id for rule in rules}) != len(rules):
        raise ConfigurationError("profile: duplicate rule ID")
    return Profile(
        data["profile"], data["policy_version"], emit, accept, labels, data["public_types"], rules
    )
