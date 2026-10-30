"""Route over LangChain / LlamaIndex tools without changing them.

`Router.route` accepts LangChain tools (`name`, `description`, `args_schema`/`func`) and LlamaIndex
tools (`metadata.name`, `metadata.description`). If those packages are installed the real classes
are used; otherwise lightweight stand-ins with the same attributes are used so this runs anywhere.
    python examples/framework_adapters.py [--model nisaralll/soro-kev-0.8b]
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from _common import router_from_args


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def search_web(query: str) -> str:
    """Search the web for a query."""
    return f"results for {query}"


def langchain_tools():
    try:
        from langchain_core.tools import StructuredTool
        return [StructuredTool.from_function(multiply), StructuredTool.from_function(search_web)], "langchain_core"
    except Exception:
        return [SimpleNamespace(name=f.__name__, description=f.__doc__, func=f) for f in (multiply, search_web)], "stand-in"


def llamaindex_tools():
    try:
        from llama_index.core.tools import FunctionTool
        return [FunctionTool.from_defaults(fn=multiply), FunctionTool.from_defaults(fn=search_web)], "llama_index"
    except Exception:
        return [SimpleNamespace(metadata=SimpleNamespace(name=f.__name__, description=f.__doc__), fn=f)
                for f in (multiply, search_web)], "stand-in"


def main():
    router, _ = router_from_args(__doc__)
    for loader in (langchain_tools, llamaindex_tools):
        tools, kind = loader()
        d = router.route("multiply 6 by 7", tools)
        result = None
        if d.should_execute:
            fn = {"multiply": multiply, "search_web": search_web}[d.tool]
            result = fn(**d.repair_args({"a": 6, "b": 7})) if d.tool == "multiply" else fn("multiply 6 by 7")
        print(json.dumps({"framework": loader.__name__, "classes": kind, "action": str(d.action), "tool": d.tool, "result": result}))


if __name__ == "__main__":
    main()
