"""Profile selection and validation do not require model or HTTP dependencies."""

import pytest

from backend.detectors.app.config import DEFAULT_PROFILE_PATH, load_profile
from backend.detectors.app.errors import ConfigurationError


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv("PROFILE_PATH", raising=False)
    monkeypatch.delenv("NER_LABELS", raising=False)


@pytest.fixture
def profile_file(tmp_path):
    path = tmp_path / "experiment.toml"
    source = DEFAULT_PROFILE_PATH.read_text().replace("pii-ru-en-v5", "experiment-v1")
    path.write_text(source)
    return path


def test_default_profile_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_profile().name == "pii-ru-en-v5"


def test_profile_selected_from_environment(profile_file, monkeypatch):
    monkeypatch.setenv("PROFILE_PATH", str(profile_file))
    selected = load_profile()
    assert selected.name == "experiment-v1"
    assert selected.accept_threshold == 0.50
    assert selected.rules[0].id == "email_ascii_v1"


def test_explicit_path_overrides_environment(profile_file, monkeypatch):
    monkeypatch.setenv("PROFILE_PATH", "")
    assert load_profile(profile_file).name == "experiment-v1"


def test_relative_profile_path_uses_working_directory(profile_file, monkeypatch):
    monkeypatch.chdir(profile_file.parent)
    monkeypatch.setenv("PROFILE_PATH", profile_file.name)
    assert load_profile().name == "experiment-v1"


def test_environment_is_read_on_each_load(profile_file, monkeypatch):
    monkeypatch.setenv("PROFILE_PATH", str(profile_file))
    assert load_profile().name == "experiment-v1"
    monkeypatch.delenv("PROFILE_PATH")
    assert load_profile().name == "pii-ru-en-v5"


def test_ner_labels_override_selected_profile(profile_file, monkeypatch):
    monkeypatch.setenv("PROFILE_PATH", str(profile_file))
    monkeypatch.setenv("NER_LABELS", "employee_id,passport")
    selected = load_profile()
    assert selected.name == "experiment-v1"
    assert selected.labels == ("employee_id", "passport")
    assert selected.public_type("employee_id") == "EMPLOYEE_ID"
    assert selected.public_type("passport") == "PASSPORT"
    assert selected.model_groups == (("employee_id", "passport"),)


@pytest.mark.parametrize(
    "groups", ["[]", "[['person']]", "[['person'], ['person']]", "'private'", "[[]]"]
)
def test_invalid_ner_groups_are_rejected(profile_file, groups):
    lines = profile_file.read_text().splitlines()
    profile_file.write_text(
        "\n".join(
            "ner_label_groups = " + groups if line.startswith("ner_label_groups =") else line
            for line in lines
        )
    )
    with pytest.raises(ConfigurationError, match=r"^profile:"):
        load_profile(profile_file)


def test_legacy_profiles_use_one_schema(profile_file):
    profile_file.write_text(
        "\n".join(
            line
            for line in profile_file.read_text().splitlines()
            if not line.startswith(("ner_label_groups =", "ner_schema_labels ="))
        )
    )
    selected = load_profile(profile_file)
    assert selected.model_groups == (selected.labels,)


@pytest.mark.parametrize(
    "schemas", ["[]", "[['unknown'],['medical_policy']]", "[['person']]", "'private'"]
)
def test_invalid_schema_context_is_rejected(profile_file, schemas):
    profile_file.write_text(
        "\n".join(
            "ner_schema_labels = " + schemas if line.startswith("ner_schema_labels =") else line
            for line in profile_file.read_text().splitlines()
        )
    )
    with pytest.raises(ConfigurationError, match=r"^profile:"):
        load_profile(profile_file)


@pytest.mark.parametrize("value", ["", " ", "\t"])
def test_empty_profile_path_is_rejected(value, monkeypatch):
    monkeypatch.setenv("PROFILE_PATH", value)
    with pytest.raises(ConfigurationError, match=r"^PROFILE_PATH: empty path$"):
        load_profile()


@pytest.mark.parametrize("directory", [False, True])
def test_unreadable_profile_has_safe_error(tmp_path, monkeypatch, directory):
    path = tmp_path / "private-path-marker.toml"
    if directory:
        path.mkdir()
    monkeypatch.setenv("PROFILE_PATH", str(path))
    with pytest.raises(ConfigurationError, match=r"^profile: cannot read file$"):
        load_profile()


@pytest.mark.parametrize("content", [b'profile = "private-content-marker"\n[broken', b"\xff"])
def test_invalid_toml_has_safe_error(tmp_path, monkeypatch, content):
    path = tmp_path / "private-path-marker.toml"
    path.write_bytes(content)
    monkeypatch.setenv("PROFILE_PATH", str(path))
    with pytest.raises(ConfigurationError, match=r"^profile: invalid TOML$"):
        load_profile()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("profile = 'experiment-v1'", "profile = 12"),
        ("policy_version = '3.0.0'", ""),
        ("default_ner_labels =", "unused_labels ="),
        ("person = 'PERSON'", "person = 12"),
        ("validator = 'email_ascii'", "validator = 12"),
    ],
)
def test_invalid_structure_is_rejected(profile_file, monkeypatch, old, new):
    profile_file.write_text(profile_file.read_text().replace(old, new))
    monkeypatch.setenv("PROFILE_PATH", str(profile_file))
    with pytest.raises(ConfigurationError, match=r"^profile: invalid structure$"):
        load_profile()


@pytest.mark.parametrize(
    ("emit", "accept"),
    [
        ("0.0", "0.50"),
        ("0.60", "0.50"),
        ("0.20", "1.1"),
        ("true", "0.50"),
        ("0.20", "false"),
        ('"private-content-marker"', "0.50"),
        ("nan", "0.50"),
    ],
)
def test_invalid_thresholds_are_rejected(profile_file, monkeypatch, emit, accept):
    source = profile_file.read_text().replace("emit_threshold = 0.20", f"emit_threshold = {emit}")
    profile_file.write_text(
        source.replace("accept_threshold = 0.50", f"accept_threshold = {accept}")
    )
    monkeypatch.setenv("PROFILE_PATH", str(profile_file))
    with pytest.raises(ConfigurationError, match=r"^profile: invalid thresholds$"):
        load_profile()


def test_invalid_defaults_are_rejected_even_with_label_override(profile_file, monkeypatch):
    profile_file.write_text(profile_file.read_text().replace("'first_name'", "'person'"))
    monkeypatch.setenv("PROFILE_PATH", str(profile_file))
    monkeypatch.setenv("NER_LABELS", "employee_id")
    with pytest.raises(ConfigurationError, match=r"^profile: invalid default NER labels$"):
        load_profile()


def test_duplicate_rule_ids_are_rejected(profile_file, monkeypatch):
    source = profile_file.read_text().replace("id = 'phone_plus_v1'", "id = 'email_ascii_v1'")
    profile_file.write_text(source)
    monkeypatch.setenv("PROFILE_PATH", str(profile_file))
    with pytest.raises(ConfigurationError, match=r"^profile: duplicate rule ID$"):
        load_profile()


def test_legacy_profile_defaults_to_original_merge_policy(profile_file):
    profile_file.write_text(profile_file.read_text().replace("resolve_conflicts = true", ""))
    assert not load_profile(profile_file).resolve_conflicts


def test_invalid_conflict_option_is_rejected(profile_file):
    profile_file.write_text(
        profile_file.read_text().replace("resolve_conflicts = true", "resolve_conflicts = 'true'")
    )
    with pytest.raises(ConfigurationError, match="invalid structure"):
        load_profile(profile_file)
