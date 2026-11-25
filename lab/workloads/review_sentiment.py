"""Review sentiment (Lambda, DIRECT to api.anthropic.com, NOT instrumented)."""

EVENTS = [
    {"review": "Absolutely love it, setup took two minutes."},
    {"review": "Support never answered my ticket. Cancelling."},
    {"review": "It's fine. Does what it says, nothing more."},
    {"review": "The new dashboard is confusing but the exports are great."},
]


def handle(rt, event):
    resp = rt.llm(
        system="Classify the review sentiment. Reply with one word: positive, negative, neutral or mixed.",
        messages=[{"role": "user", "content": event["review"]}],
        max_tokens=50,
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()
