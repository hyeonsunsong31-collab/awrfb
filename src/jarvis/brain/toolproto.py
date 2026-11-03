"""Prompted tool calling for models without native function calling (some local models).

Tools are described in the system prompt; the model answers with a fenced JSON block
```tool {"name": "...", "arguments": {...}} ```. We parse those back into ToolCall objects.
"""

from __future__ import annotations

import json
import re
import uuid

from jarvis.brain.types import ToolCall

_BLOCK = re.compile(r"```(?:tool|json)?\s*(\{.*?\})\s*```", re.S)


def render_tools_prompt(tools: list[dict]) -> str:
    lines = [
        "You can use tools. To call one, reply with ONLY a fenced block like:",
        '```tool\n{"name": "tool_name", "arguments": {"arg": "value"}}\n```',
        "You may output several blocks to call several tools. Otherwise answer normally.",
        "Available tools:",
    ]
    for t in tools:
        f = t.get("function", t)
        lines.append(f"- {f['name']}: {f.get('description', '')}")
        lines.append(f"  parameters: {json.dumps(f.get('parameters', {}), ensure_ascii=False)}")
    return "\n".join(lines)


def parse_prompted_tool_calls(text: str, known: set[str] | None = None) -> tuple[list[ToolCall], str]:
    """Returns (tool calls, remaining text with the blocks removed)."""
    calls: list[ToolCall] = []
    for m in _BLOCK.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except ValueError:
            continue
        name = obj.get("name") or obj.get("tool")
        if not isinstance(name, str) or (known is not None and name not in known):
            continue
        args = obj.get("arguments", obj.get("args", {}))
        calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}", name, json.dumps(args, ensure_ascii=False)))
    remaining = _BLOCK.sub("", text).strip() if calls else text
    return calls, remaining
