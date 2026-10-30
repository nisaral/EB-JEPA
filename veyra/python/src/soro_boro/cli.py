"""``soro-boro`` command line: route, eval, serve, doctor, convert-head."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from soro_boro.router import DEFAULT_MODEL, Router, RouterLoadError


def _load_json(path: str) -> Any:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"error: file not found: {path}")
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _tools_from_file(path: str) -> Any:
    data = _load_json(path)
    if isinstance(data, dict) and "tools" in data:  # MCP tools/list result or {"tools": [...]}
        return data["tools"]
    return data


def _make_router(args: argparse.Namespace) -> Router:
    if args.heuristic:
        return Router.heuristic(q_hat=args.q_hat)
    try:
        return Router.from_pretrained(args.model, device=args.device, q_hat=args.q_hat)
    except RouterLoadError as exc:
        raise SystemExit(f"error: {exc}") from None


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"HF repo id or local dir (default {DEFAULT_MODEL})")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda (default auto)")
    p.add_argument("--q-hat", type=float, default=None, help="override the checkpoint's calibrated q_hat")
    p.add_argument("--heuristic", action="store_true", help="no weights: lexical fallback scorer (demo/testing)")


def cmd_route(args: argparse.Namespace) -> int:
    router = _make_router(args)
    tools = _tools_from_file(args.tools)
    d = router.route(args.query, tools=tools)
    out = d.to_dict()
    if args.args:
        out["repaired_args"] = d.repair_args(json.loads(args.args)) if d.tool else None
    print(json.dumps(out, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Rows: {"query": ..., "expected": tool|null|"__abstain__", "tools"?: [...]}"""
    router = _make_router(args)
    rows = _load_json(args.data)
    default_tools = _tools_from_file(args.tools) if args.tools else None
    n = correct = executed = exec_correct = abstain_rows = abstain_ok = 0
    for row in rows:
        tools = row.get("tools", default_tools)
        if tools is None:
            raise SystemExit("error: row has no 'tools' and no --tools file was given")
        d = router.route(row["query"], tools=tools)
        exp = row.get("expected")
        exp = None if exp in (None, "__abstain__", "") else exp
        n += 1
        top = d.tool if d.action.value != "ABSTAIN" else None
        correct += int(top == exp)
        if exp is None:
            abstain_rows += 1
            abstain_ok += int(d.action.value == "ABSTAIN")
        if d.action.value == "EXECUTE":
            executed += 1
            exec_correct += int(d.tool == exp)
    report = {
        "rows": n,
        "top1_accuracy": correct / n if n else None,
        "execute_rate": executed / n if n else None,
        "execute_precision": exec_correct / executed if executed else None,
        "no_tool_rows": abstain_rows,
        "no_tool_abstain_rate": abstain_ok / abstain_rows if abstain_rows else None,
        "q_hat": router.q_hat,
        "backend": router.backend,
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - blocking
    from soro_boro.server import serve

    serve(_make_router(args), host=args.host, port=args.port)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    import importlib
    import platform

    import soro_boro

    lines = [f"soro-boro {soro_boro.__version__}", f"python {platform.python_version()} ({sys.platform})"]
    ok = True
    for mod, why in (("torch", "required"), ("transformers", "required"), ("peft", "required"),
                     ("safetensors", "required"), ("huggingface_hub", "required"),
                     ("fastapi", "optional: serve"), ("uvicorn", "optional: serve")):
        try:
            m = importlib.import_module(mod)
            lines.append(f"  ok   {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            lines.append(f"  MISS {mod} ({why})")
            ok = ok and why != "required"
    try:
        import torch

        lines.append(f"  cuda available: {torch.cuda.is_available()}")
    except ImportError:
        pass
    if args.check_model:
        try:
            r = Router.from_pretrained(args.model, device=args.device)
            lines.append(f"  model {args.model}: loaded ({r.backend}, q_hat={r.q_hat:.4f})")
        except RouterLoadError as exc:
            ok = False
            lines.append(f"  model {args.model}: FAILED\n    " + str(exc).replace("\n", "\n    "))
    print("\n".join(lines))
    return 0 if ok else 1


def cmd_convert_head(args: argparse.Namespace) -> int:
    from soro.routing.kev_model import convert_head_pt_to_safetensors

    print(convert_head_pt_to_safetensors(args.checkpoint_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="soro-boro", description="Calibrated tool routing with Soro-Kev.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("route", help="route one query")
    p.add_argument("query")
    p.add_argument("--tools", required=True, help="JSON file: OpenAI/MCP/dict tool list (or {'tools': [...]})")
    p.add_argument("--args", help="JSON of proposed arguments to repair against the chosen tool")
    _add_model_args(p)
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("eval", help="evaluate on a JSON/JSONL file of {query, expected, tools?}")
    p.add_argument("data")
    p.add_argument("--tools", help="tool list used for rows without their own 'tools'")
    _add_model_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="run the HTTP server (pip install 'soro-boro[server]')")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    _add_model_args(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("doctor", help="check the environment (and optionally load the model)")
    p.add_argument("--check-model", action="store_true")
    _add_model_args(p)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("convert-head", help="convert a checkpoint's head.pt to head.safetensors")
    p.add_argument("checkpoint_dir")
    p.set_defaults(func=cmd_convert_head)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
