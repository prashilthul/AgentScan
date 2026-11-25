"""Orders CRUD API (Lambda). No AI - shares an execution role with ticket-summarizer."""

import random

EVENTS = [{"op": "get", "id": f"A-{1000 + i}"} for i in range(4)] + [{"op": "put", "id": "A-2000"}]


def handle(rt, event):
    with rt.span(
        f"dynamodb.{'GetItem' if event['op'] == 'get' else 'PutItem'}",
        **{"db.system": "dynamodb", "aws.dynamodb.table_names": "orders"},
    ):
        rt.log(f"{event['op']} order {event['id']}")
        return {"id": event["id"], "status": random.choice(["paid", "shipped"])}
