"""Normalize tool definitions from common ecosystems into one ``ToolSpec``.

Accepted inputs:

* OpenAI / Chat Completions: ``{"type": "function", "function": {"name", "description", "parameters"}}``
* OpenAI Responses / flat function: ``{"type": "function", "name", "description", "parameters"}``
* MCP ``tools/list`` entries: ``{"name", "description", "inputSchema"}`` (or objects with those attributes)
* Anthropic: ``{"name", "description", "input_schema"}``
* LangChain / LlamaIndex tools: objects with ``name`` / ``description`` (and ``args_schema`` / ``metadata``)
* Plain dicts ``{"name", "description", "parameters"?}``, ``(name, description)`` tuples,
  ``{name: description}`` mappings, and Python callables (docstring + signature).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

__all__ = ["ToolSpec", "normalize_tool", "normalize_tools", "ToolSchemaError"]


class ToolSchemaError(ValueError):
    """Raised when a tool definition cannot be understood."""


@dataclass(frozen=True)
class ToolSpec:
    """A tool as the router sees it."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON Schema object
    source: str = "dict"
    raw: Any = field(default=None, repr=False, compare=False)

    @property
    def properties(self) -> dict[str, Any]:
        return dict(self.parameters.get("properties") or {})

    @property
    def required(self) -> list[str]:
        return list(self.parameters.get("required") or [])

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _schema_from_callable(fn: Callable[..., Any]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    required: list[str] = []
    try:
        import typing

        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    for pname, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or pname in ("self", "cls"):
            continue
        ann = hints.get(pname, p.annotation)
        if isinstance(ann, str):
            ann = {"str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict}.get(ann, str)
        entry: dict[str, Any] = {"type": _PY_TO_JSON.get(ann, "string")}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            entry["default"] = p.default
        props[pname] = entry
    return {"type": "object", "properties": props, "required": required}


def _schema_from_pydantic(model: Any) -> dict[str, Any]:
    for attr in ("model_json_schema", "schema"):
        fn = getattr(model, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:  # pragma: no cover - defensive
                pass
    return {}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_tool(tool: Any) -> ToolSpec:
    """Convert one tool definition into a :class:`ToolSpec`."""
    if isinstance(tool, ToolSpec):
        return tool
    if isinstance(tool, tuple) and len(tool) == 2 and isinstance(tool[0], str):
        return ToolSpec(name=tool[0], description=str(tool[1]), source="tuple", raw=tool)

    if isinstance(tool, Mapping):
        fn = tool.get("function")
        if tool.get("type") == "function" and isinstance(fn, Mapping):  # OpenAI chat
            return ToolSpec(
                name=str(fn["name"]),
                description=str(fn.get("description") or ""),
                parameters=dict(fn.get("parameters") or {}),
                source="openai",
                raw=tool,
            )
        if "name" not in tool:
            raise ToolSchemaError(f"tool definition has no 'name': {tool!r}")
        for key, source in (("inputSchema", "mcp"), ("input_schema", "anthropic"), ("parameters", "dict")):
            if key in tool:
                params, src = dict(tool.get(key) or {}), source
                break
        else:
            params, src = {}, "dict"
        if tool.get("type") == "function":
            src = "openai"
        return ToolSpec(str(tool["name"]), str(tool.get("description") or ""), params, src, tool)

    name = _get(tool, "name")
    if isinstance(name, str) and name:
        # MCP SDK Tool objects, LangChain BaseTool, LlamaIndex FunctionTool (via .metadata)
        desc = _get(tool, "description") or ""
        params: dict[str, Any] = {}
        src = "object"
        if _get(tool, "inputSchema") is not None:
            params, src = dict(_get(tool, "inputSchema")), "mcp"
        elif _get(tool, "args_schema") is not None:
            params, src = _schema_from_pydantic(_get(tool, "args_schema")), "langchain"
        elif callable(_get(tool, "func")):
            params, src = _schema_from_callable(_get(tool, "func")), "langchain"
        return ToolSpec(name, str(desc), params, src, tool)

    meta = _get(tool, "metadata")
    if meta is not None and isinstance(_get(meta, "name"), str):  # LlamaIndex
        params = {}
        fn_schema = _get(meta, "fn_schema")
        if fn_schema is not None:
            params = _schema_from_pydantic(fn_schema)
        return ToolSpec(_get(meta, "name"), str(_get(meta, "description") or ""), params, "llamaindex", tool)

    if callable(tool) and hasattr(tool, "__name__"):
        doc = inspect.getdoc(tool) or ""
        return ToolSpec(tool.__name__, doc.split("\n\n")[0], _schema_from_callable(tool), "callable", tool)

    raise ToolSchemaError(f"unsupported tool definition of type {type(tool).__name__}: {tool!r}")


def normalize_tools(tools: Iterable[Any] | Mapping[str, Any] | None) -> list[ToolSpec]:
    """Normalize a list (or ``{name: description}`` mapping) of tools; names must be unique."""
    if tools is None:
        return []
    if isinstance(tools, Mapping) and not ("name" in tools or "function" in tools):
        items: list[Any] = [
            (k, v) if isinstance(v, str) else {"name": k, **dict(v)} for k, v in tools.items()
        ]
    else:
        items = list(tools)
    specs = [normalize_tool(t) for t in items]
    seen: set[str] = set()
    for s in specs:
        if s.name in seen:
            raise ToolSchemaError(f"duplicate tool name: {s.name!r}")
        if s.name == "__abstain__":
            raise ToolSchemaError("'__abstain__' is reserved for the abstain option")
        seen.add(s.name)
    return specs
