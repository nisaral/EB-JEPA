"""A tiny MCP-style gateway: route a request across tools from several MCP servers.

Tool definitions use the MCP `tools/list` shape (name, description, inputSchema). The gateway routes,
repairs the proposed arguments against the chosen tool's inputSchema, and returns the `tools/call`
it *would* send. No MCP server is contacted.
    python examples/mcp_gateway.py [--model nisaralll/soro-kev-0.8b]
"""
from __future__ import annotations

import json

from _common import router_from_args

SERVERS = {
    "github": [
        {"name": "create_issue", "description": "Create a GitHub issue in a repository",
         "inputSchema": {"type": "object", "properties": {"repo": {"type": "string"}, "title": {"type": "string"}}, "required": ["repo", "title"]}},
        {"name": "list_pull_requests", "description": "List open pull requests for a repository",
         "inputSchema": {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]}},
    ],
    "filesystem": [
        {"name": "read_file", "description": "Read a file from disk by path",
         "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    ],
}


def gateway(router, query, proposed_args=None):
    tools, owner = [], {}
    for server, ts in SERVERS.items():
        for t in ts:
            tools.append(t)
            owner[t["name"]] = server
    d = router.route(query, tools)
    out = {"query": query, "action": str(d.action), "candidates": d.candidates}
    if d.should_execute:
        out["call"] = {"server": owner[d.tool], "method": "tools/call",
                       "params": {"name": d.tool, "arguments": d.repair_args(proposed_args or {})}}
    return out


def main():
    router, _ = router_from_args(__doc__)
    print(json.dumps(gateway(router, "list open pull requests in nisaral/Soro-Boro", {"Repo": "nisaral/Soro-Boro"})))
    print(json.dumps(gateway(router, "read the file at /etc/hosts", {"file_path": "/etc/hosts"})))


if __name__ == "__main__":
    main()
