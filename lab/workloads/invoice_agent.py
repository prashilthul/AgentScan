"""Invoice approval agent (Lambda, via Bedrock, NOT instrumented with OpenTelemetry)."""

from lab.workloads._loop import run_agent, tool

VENDORS = {
    "V-77": {"name": "Paperco", "trusted": True},
    "V-13": {"name": "Shady LLC", "trusted": False},
}
POS = {"PO-5": {"vendor": "V-77", "amount": 1200.0}, "PO-9": {"vendor": "V-13", "amount": 300.0}}

TOOLS = [
    tool("get_vendor", "Vendor details by id.", {"vendor_id": {"type": "string"}}, ["vendor_id"]),
    tool("get_po", "Purchase order by id.", {"po_id": {"type": "string"}}, ["po_id"]),
    tool(
        "decide",
        "Record the decision.",
        {"decision": {"type": "string", "enum": ["approve", "flag"]}, "reason": {"type": "string"}},
        ["decision", "reason"],
    ),
]
IMPLS = {
    "get_vendor": lambda vendor_id: VENDORS.get(vendor_id, {"error": "unknown"}),
    "get_po": lambda po_id: POS.get(po_id, {"error": "unknown"}),
    "decide": lambda decision, reason: {"recorded": decision},
}
SYSTEM = (
    "You approve invoices. Check the vendor and the purchase order with the tools, "
    "then record a decision with `decide`. Approve only trusted vendors whose amount "
    "matches the PO."
)

EVENTS = [
    {"invoice": "Invoice INV-1 from V-77 for $1200 against PO-5"},
    {"invoice": "Invoice INV-2 from V-13 for $900 against PO-9"},
    {"invoice": "Invoice INV-3 from V-77 for $1500 against PO-5"},
]


def handle(rt, event):
    return run_agent(rt, "invoice-agent", SYSTEM, event["invoice"], TOOLS, IMPLS)
