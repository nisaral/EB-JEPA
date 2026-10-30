"""Pytest configuration and compatibility aliases for Soro.

Provides transparent backwards compatibility so legacy tests importing from
`veyra.*` seamlessly resolve to the exact identical objects in `soro.*`.
"""

import importlib
import importlib.machinery
import sys


class AliasLoader:
    def __init__(self, target: str):
        self.target = target

    def create_module(self, spec):
        mod = importlib.import_module(self.target)
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):
        pass


class VeyraAliasFinder:
    """Redirects all `veyra.*` module imports to identical `soro.*` instances."""

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname == "veyra" or fullname.startswith("veyra."):
            soro_name = "soro" + fullname[5:]
            return importlib.machinery.ModuleSpec(fullname, AliasLoader(soro_name))
        return None


if not any(isinstance(f, VeyraAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, VeyraAliasFinder())

# Shared isolation/skip hooks (see repo_test_support.py)
import importlib.util as _ilu
import pathlib as _pl

_p = next(p / "repo_test_support.py" for p in _pl.Path(__file__).resolve().parents if (p / "repo_test_support.py").is_file())
_spec = _ilu.spec_from_file_location("repo_test_support", _p)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pytest_collection_modifyitems = _mod.pytest_collection_modifyitems
_isolated_soro_boro_state = _mod._isolated_soro_boro_state
