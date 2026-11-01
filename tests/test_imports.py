import importlib

import pytest


@pytest.mark.parametrize("module", ["detectors", "agent", "evaluation"])
def test_package_imports(module):
    importlib.import_module(module)
