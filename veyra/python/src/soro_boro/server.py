"""HTTP server for the router (``soro-boro serve``; needs ``pip install soro-boro[server]``).

Endpoints:

* ``GET  /healthz``             -> ``{"ok": true, "backend": ..., "q_hat": ...}``
* ``POST /route``               -> body ``{"query": str, "tools": [...], "q_hat"?: float}``, returns a RoutingDecision dict
* ``POST /v1/route``            -> OpenAI-shaped body ``{"messages": [...], "tools": [...]}``; routes the last user
  message and returns the decision plus a ``tool_choice`` you can forward to an OpenAI-compatible API
  (``{"type": "function", "function": {"name": ...}}`` on EXECUTE, ``"none"`` on ABSTAIN, ``"auto"`` on ASK).
"""
from __future__ import annotations

from typing import Any

from soro_boro.router import Router


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):  # content parts
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def tool_choice_for(decision: Any) -> Any:
    if decision.action.value == "EXECUTE":
        return {"type": "function", "function": {"name": decision.tool}}
    return "none" if decision.action.value == "ABSTAIN" else "auto"


def create_app(router: Router):
    try:
        from fastapi import Body, FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the HTTP server needs FastAPI: pip install 'soro-boro[server]'") from exc

    from soro_boro.tool_schema import ToolSchemaError

    app = FastAPI(title="Soro-Boro router", version=__import__("soro_boro").__version__)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "backend": router.backend, "q_hat": router.q_hat}

    def _route(query: str, tools: Any, q_hat: float | None) -> Any:
        try:
            return router.route(query, tools=tools, q_hat=q_hat)
        except (ToolSchemaError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/route")
    def route(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if "query" not in body:
            raise HTTPException(status_code=400, detail="missing 'query'")
        return _route(body["query"], body.get("tools") or [], body.get("q_hat")).to_dict()

    @app.post("/v1/route")
    def route_openai(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        query = _last_user_text(body.get("messages") or [])
        if not query:
            raise HTTPException(status_code=400, detail="no user message found in 'messages'")
        d = _route(query, body.get("tools") or [], body.get("q_hat"))
        return {"decision": d.to_dict(), "tool_choice": tool_choice_for(d)}

    return app


def serve(router: Router, host: str = "127.0.0.1", port: int = 8765) -> None:  # pragma: no cover
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("the HTTP server needs uvicorn: pip install 'soro-boro[server]'") from exc
    uvicorn.run(create_app(router), host=host, port=port)
