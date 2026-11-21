import importlib
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "backend",
        "backend.api",
        "backend.schemas",
        "backend.detectors",
        "backend.detectors.app.config",
        "backend.detectors.app.regex_detector",
        "backend.detectors.scripts.acceptance",
        "backend.agent",
        "evaluation",
    ],
)
def test_package_imports(module):
    importlib.import_module(module)


def test_project_scripts_are_not_shadowed_by_detector_scripts():
    spec = importlib.util.find_spec("scripts")
    project_scripts = Path(__file__).resolve().parents[1] / "scripts"
    assert spec is not None and spec.submodule_search_locations is not None
    assert project_scripts in [Path(path).resolve() for path in spec.submodule_search_locations]
