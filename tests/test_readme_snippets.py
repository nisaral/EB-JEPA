"""The README quickstart and docs/quickstart.md python blocks must run as written (offline lines only)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]



def test_readme_quickstart_runs(capsys):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    block = re.findall(r"```python\n(.*?)```", text, re.S)[0]
    assert "Router.heuristic()" in block
    exec(compile(block, "README.md", "exec"), {})
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("EXECUTE get_weather")
    assert out[1] == "{'city': 'Paris'}"


def _downloads(block):
    """Blocks that load real weights are skipped (covered by the opt-in pretrained smoke test)."""
    return any("from_pretrained(" in ln.split("#")[0] for ln in block.splitlines())


@pytest.mark.parametrize("path", ["docs/quickstart.md", "docs/API.md"])
def test_docs_python_blocks_run(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    blocks = [b for b in re.findall(r"```python\n(.*?)```", text, re.S) if not _downloads(b)]
    assert blocks
    ns: dict = {}
    for b in blocks:  # blocks in a page share state, like a notebook
        exec(compile(b, path, "exec"), ns)
