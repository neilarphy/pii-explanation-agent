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
    resolve_conflicts: bool = False
    ner_label_groups: tuple[tuple[str, ...], ...] = ()
    ner_schema_labels: tuple[tuple[str, ...], ...] = ()

    @property
    def model_groups(self) -> tuple[tuple[str, ...], ...]:
        groups = self.ner_label_groups
        # Programmatic label overrides (e.g. dataclasses.replace) also stay coherent.
        return (
            groups if tuple(label for g in groups for label in g) == self.labels else (self.labels,)
        )

    @property
    def model_schemas(self) -> tuple[tuple[str, ...], ...]:
        return (
            self.ner_schema_labels
            if self.model_groups == self.ner_label_groups and self.ner_schema_labels
            else self.model_groups
        )

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
        resolve_conflicts = data.get("resolve_conflicts", False)
        if (
            not all(isinstance(value, str) and value.strip() for value in (name, policy))
            or not isinstance(defaults, list)
            or not isinstance(public_types, dict)
            or not all(
                isinstance(key, str) and isinstance(value, str) and key and value.strip()
                for key, value in public_types.items()
            )
            or not isinstance(data["rules"], list)
            or type(resolve_conflicts) is not bool
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
        raw_groups = data.get("ner_label_groups", [list(defaults)])
        if not isinstance(raw_groups, list) or not all(isinstance(g, list) for g in raw_groups):
            raise ValueError
        groups = tuple(parse_labels(None, tuple(g)) for g in raw_groups)
        if tuple(label for g in groups for label in g) != defaults:
            raise ValueError
        raw_schemas = data.get("ner_schema_labels", raw_groups)
        if not isinstance(raw_schemas, list) or not all(isinstance(g, list) for g in raw_schemas):
            raise ValueError
        schemas = tuple(parse_labels(None, tuple(g)) for g in raw_schemas)
        if len(schemas) != len(groups) or any(
            not set(group) <= set(schema) <= set(defaults)
            for group, schema in zip(groups, schemas, strict=True)
        ):
            raise ValueError
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
    if "NER_LABELS" in os.environ:
        groups = (labels,)
        schemas = groups
    if len({rule.id for rule in rules}) != len(rules):
        raise ConfigurationError("profile: duplicate rule ID")
    return Profile(
        name, policy, emit, accept, labels, public_types, rules, resolve_conflicts, groups, schemas
    )
