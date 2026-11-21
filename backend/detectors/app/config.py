import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = ROOT / "config/profile.toml"
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


def load_profile(path: Path | None = None) -> Profile:
    """Select an explicit path, PROFILE_PATH, or the bundled profile, in that order."""
    if path is None:
        configured = os.environ.get("PROFILE_PATH")
        if configured is not None and not configured.strip():
            raise ConfigurationError("PROFILE_PATH: empty path")
        path = Path(configured) if configured is not None else DEFAULT_PROFILE_PATH
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError:
        raise ConfigurationError("profile: cannot read file") from None
    except ValueError:
        raise ConfigurationError("profile: invalid TOML") from None

    try:
        name, policy = data["profile"], data["policy_version"]
        emit, accept = data["emit_threshold"], data["accept_threshold"]
        defaults, public_types = data["default_ner_labels"], data["public_types"]
        if (
            not all(isinstance(value, str) and value.strip() for value in (name, policy))
            or not isinstance(defaults, list)
            or not isinstance(public_types, dict)
            or not all(
                isinstance(key, str) and isinstance(value, str) and key and value.strip()
                for key, value in public_types.items()
            )
            or not isinstance(data["rules"], list)
        ):
            raise ValueError
        rules = tuple(Rule(**rule) for rule in data["rules"])
        if not all(
            isinstance(value, str) and value.strip()
            for rule in rules
            for value in (rule.id, rule.type, rule.validator)
        ) or not all(isinstance(rule.pattern, str) for rule in rules):
            raise ValueError
        defaults = parse_labels(None, tuple(defaults))
    except ConfigurationError:
        raise ConfigurationError("profile: invalid default NER labels") from None
    except (KeyError, TypeError, ValueError):
        raise ConfigurationError("profile: invalid structure") from None
    if (
        type(emit) not in (int, float)
        or type(accept) not in (int, float)
        or not 0 < emit <= accept <= 1
    ):
        raise ConfigurationError("profile: invalid thresholds")
    labels = parse_labels(os.environ.get("NER_LABELS"), defaults)
    if len({rule.id for rule in rules}) != len(rules):
        raise ConfigurationError("profile: duplicate rule ID")
    return Profile(name, policy, emit, accept, labels, public_types, rules)
