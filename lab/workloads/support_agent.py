"""Customer-support agent (Lambda, via Bedrock): looks things up, decides, may act."""

from lab.workloads._loop import run_agent, tool

ORDERS = {
    "A-1001": {"status": "delivered", "total": 49.0, "days_since_delivery": 3},
    "A-1002": {"status": "in_transit", "total": 120.0, "days_since_delivery": None},
    "A-1003": {"status": "delivered", "total": 15.5, "days_since_delivery": 45},
}
REFUNDS: list[str] = []

TOOLS = [
    tool(
        "lookup_order",
        "Get an order's status and total.",
        {"order_id": {"type": "string"}},
        ["order_id"],
    ),
    tool("refund_policy", "Return the refund policy text.", {}, []),
    tool(
        "issue_refund",
        "Refund an order. Only if policy allows.",
        {"order_id": {"type": "string"}, "reason": {"type": "string"}},
        ["order_id", "reason"],
    ),
]


def _lookup(order_id: str):
    return ORDERS.get(order_id, {"error": "unknown order"})


def _refund(order_id: str, reason: str):
    REFUNDS.append(order_id)
    return {"refunded": order_id, "reason": reason}


IMPLS = {
    "lookup_order": _lookup,
    "refund_policy": lambda: "Refunds allowed within 30 days of delivery.",
    "issue_refund": _refund,
}

SYSTEM = (
    "You are a support agent. Use the tools to check facts before answering. "
    "Issue a refund only when the policy allows it. Reply in two sentences."
)

EVENTS = [
    {"message": "Order A-1001 arrived damaged, can I get my money back?"},
    {"message": "Where is my order A-1002?"},
    {"message": "I want a refund for A-1003, it broke."},
]


def handle(rt, event):
    return run_agent(rt, "support-agent", SYSTEM, event["message"], TOOLS, IMPLS)
