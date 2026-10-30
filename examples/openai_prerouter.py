"""Pre-route an OpenAI chat request: decide the tool *before* calling the LLM.

EXECUTE -> force `tool_choice` to that one function (and send only its schema: fewer tokens).
ASK     -> send only the candidate tools and let the LLM ask/choose.
ABSTAIN -> send no tools (plain chat answer).

Runs offline: it builds the request payload and prints it; it never calls OpenAI.
    python examples/openai_prerouter.py                # heuristic scorer, no weights
    python examples/openai_prerouter.py --model nisaralll/soro-kev-0.8b
"""
from __future__ import annotations

import json

from _common import router_from_args

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Send an email to a recipient",
     "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to"]}}},
    {"type": "function", "function": {"name": "create_calendar_event", "description": "Create a calendar event at a time",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "start": {"type": "string"}}}}},
]


def build_request(router, messages, tools=TOOLS, model="gpt-4o-mini"):
    query = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    d = router.route(query, tools)
    req = {"model": model, "messages": messages}
    by_name = {t["function"]["name"]: t for t in tools}
    if d.action == "EXECUTE":
        req["tools"] = [by_name[d.tool]]
        req["tool_choice"] = {"type": "function", "function": {"name": d.tool}}
    elif d.action == "ASK":
        req["tools"] = [by_name[n] for n in d.candidates if n in by_name]
        req["tool_choice"] = "auto"
    return d, req


def main():
    router, _ = router_from_args(__doc__)
    for q in ["What's the weather in Paris?", "Email bob@example.com that I'm late"]:
        d, req = build_request(router, [{"role": "user", "content": q}])
        print(json.dumps({"query": q, "action": str(d.action), "tool": d.tool,
                          "tool_choice": req.get("tool_choice"), "n_tools_sent": len(req.get("tools", []))}))


if __name__ == "__main__":
    main()
