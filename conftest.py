"""Root conftest: see repo_test_support.py."""
import importlib.util as _ilu
import pathlib as _pl

_p = next(p / "repo_test_support.py" for p in _pl.Path(__file__).resolve().parents if (p / "repo_test_support.py").is_file())
_spec = _ilu.spec_from_file_location("repo_test_support", _p)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pytest_collection_modifyitems = _mod.pytest_collection_modifyitems
_isolated_soro_boro_state = _mod._isolated_soro_boro_state
