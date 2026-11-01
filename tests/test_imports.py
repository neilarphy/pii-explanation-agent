import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "backend",
        "backend.api",
        "backend.schemas",
        "backend.detectors",
        "backend.agent",
        "evaluation",
    ],
)
def test_package_imports(module):
    importlib.import_module(module)
