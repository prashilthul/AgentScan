"""Ops copilot, deployed as a Bedrock AgentCore runtime.

Locally the loop runs here. What AWS sees: the runtime resource, InvokeAgentRuntime events made
by the on-call engineers' console role, and the runtime role's own Bedrock model calls.
"""

import uuid

from lab.workloads._loop import run_agent, tool

ALARMS = {"checkout-api": "p99 latency > 2s for 10 min", "search": "OK"}
RUNBOOKS = {"checkout-api": "1) check DB connections 2) restart service if pool exhausted"}

TOOLS = [
    tool("list_alarms", "List alarm state per service.", {}, []),
    tool(
        "get_runbook",
        "Get the runbook for a service.",
        {"service": {"type": "string"}},
        ["service"],
    ),
    tool(
        "restart_service",
        "Restart a service (simulated).",
        {"service": {"type": "string"}},
        ["service"],
    ),
]
IMPLS = {
    "list_alarms": lambda: ALARMS,
    "get_runbook": lambda service: RUNBOOKS.get(service, "no runbook"),
    "restart_service": lambda service: {"restarted": service},
}
SYSTEM = "You are an SRE assistant. Investigate with the tools and act per the runbook. Be brief."

EVENTS = [
    {"message": "Anything on fire right now?"},
    {"message": "Checkout is slow, please handle it."},
    {"message": "What's the runbook for checkout-api?"},
]


def handle(rt, event):
    rt.mirror.agent_invoked(
        rt.mirror.assumed_role("ops-console", "oncall-engineer"),
        rt.arn,
        session_id=str(uuid.uuid4()),
    )
    return run_agent(rt, "ops-copilot", SYSTEM, event["message"], TOOLS, IMPLS)
