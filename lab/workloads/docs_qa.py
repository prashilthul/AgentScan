"""Docs Q&A (ECS, via Bedrock): retrieval-augmented, ONE model call per question.

Retrieval is ordinary code the service always runs - the model never chooses it - so this
is a single-call LLM app, not an agent.
"""

DOCS = {
    "billing.md": "Invoices are issued on the 1st. Annual plans get a 15% discount.",
    "sso.md": "SSO supports SAML and OIDC. Rotate IdP certificates in Settings > Security.",
    "export.md": "Dashboards export to CSV. PDF export is on the roadmap.",
}

EVENTS = [
    {"question": "How do I rotate the SSO certificate?"},
    {"question": "Is there a discount for annual billing?"},
    {"question": "Can I export dashboards to PDF?"},
]


def _retrieve(q: str) -> list[str]:
    words = {w.lower().strip("?") for w in q.split()}
    return [
        f"{k}: {v}" for k, v in DOCS.items() if words & {w.lower().strip(".,>") for w in v.split()}
    ]


def handle(rt, event):
    with rt.span("retrieve_documents", **{"retrieval.top_k": 3}):
        context = _retrieve(event["question"]) or list(DOCS.values())
    resp = rt.llm(
        system="Answer only from the provided documents. One or two sentences.",
        messages=[
            {
                "role": "user",
                "content": "Documents:\n"
                + "\n".join(context)
                + f"\n\nQuestion: {event['question']}",
            }
        ],
        max_tokens=300,
    )
    return "".join(b.text for b in resp.content if b.type == "text")
