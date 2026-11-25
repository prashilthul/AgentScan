"""Ticket summariser (Lambda, via Bedrock): exactly one model call per ticket."""

EVENTS = [
    {
        "ticket": "Customer says the app logs them out every 5 minutes on Android 14 since update 4.2."
    },
    {"ticket": "Invoice #8812 charged twice; customer wants one charge reversed. Card ends 4421."},
    {"ticket": "Feature request: export dashboards to PDF with company branding."},
    {"ticket": "SSO login fails with 'invalid audience' after the IdP certificate rotation."},
]


def handle(rt, event):
    with rt.span(
        "dynamodb.GetItem", **{"db.system": "dynamodb", "aws.dynamodb.table_names": "tickets"}
    ):
        pass
    resp = rt.llm(
        system="Summarise the support ticket in one line: <category>: <summary>.",
        messages=[{"role": "user", "content": event["ticket"]}],
        max_tokens=200,
    )
    return "".join(b.text for b in resp.content if b.type == "text")
