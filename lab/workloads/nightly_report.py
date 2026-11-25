"""Nightly report job (Lambda, not instrumented). Holds bedrock:* but never calls a model."""

EVENTS = [{"date": "2026-11-23"}, {"date": "2026-11-24"}, {"date": "2026-11-25"}]


def handle(rt, event):
    rows = [{"day": event["date"], "orders": 120 + i, "refunds": i % 4} for i in range(5)]
    rt.log(f"report {event['date']}: {sum(r['orders'] for r in rows)} orders aggregated")
    return {"rows": len(rows)}
