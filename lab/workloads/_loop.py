"""A plain manual tool-use loop, shared by the agent workloads.

The model decides which tool to call; the tool result is fed back; repeat until the model
stops asking for tools. That feedback loop is what makes a workload an agent.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from lab.shim.runtime import Runtime


def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def run_agent(
    rt: Runtime,
    agent_name: str,
    system: str,
    task: str,
    tools: list[dict],
    impls: dict[str, Callable[..., object]],
    max_steps: int = 6,
) -> str:
    messages: list[dict] = [{"role": "user", "content": task}]
    with rt.agent_span(agent_name):
        for _ in range(max_steps):
            resp = rt.llm(system=system, tools=tools, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text")
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                with rt.tool_span(block.name):
                    try:
                        out = impls[block.name](**block.input)
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(out),
                            }
                        )
                    except Exception as e:  # noqa: BLE001 - report tool errors to the model
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"error: {e}",
                                "is_error": True,
                            }
                        )
            messages.append({"role": "user", "content": results})
        return "(stopped: step limit)"
