"""Invoice field extractor (Lambda, via Bedrock).

It offers a strict tool schema only to get structured JSON back. The "tool" is never run and
no result is fed back to the model: one call per document. An easy one to mislabel as agent.
"""

from lab.workloads._loop import tool

RECORD = tool(
    "record_invoice",
    "Record the fields extracted from an invoice.",
    {
        "vendor": {"type": "string"},
        "invoice_number": {"type": "string"},
        "total": {"type": "number"},
        "currency": {"type": "string"},
    },
    ["vendor", "invoice_number", "total", "currency"],
) | {"strict": True}

EVENTS = [
    {"text": "PAPERCO LTD - Invoice INV-2231 - Total due: EUR 1,240.00"},
    {"text": "Bill from Northwind Traders, no. 7781, amount $89.90 USD"},
    {"text": "Invoice 55-A / Globex Corp / total GBP 410.25"},
]


def handle(rt, event):
    with rt.span("s3.GetObject", **{"aws.s3.bucket": "acme-invoices"}):
        text = event["text"]
    resp = rt.llm(
        system="Extract the invoice fields by calling record_invoice exactly once.",
        tools=[RECORD],
        messages=[{"role": "user", "content": text}],
        max_tokens=400,
    )
    calls = [b.input for b in resp.content if b.type == "tool_use"]
    return calls[0] if calls else None
