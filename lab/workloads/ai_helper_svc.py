"""'ai-helper-svc' (ECS): despite the name, a feature-flag lookup service. No AI."""

FLAGS = {"new-dashboard": True, "pdf-export": False, "ai-summaries": True}

EVENTS = [{"flag": f} for f in FLAGS] + [{"flag": "unknown-flag"}]


def handle(rt, event):
    with rt.span(
        "dynamodb.GetItem", **{"db.system": "dynamodb", "aws.dynamodb.table_names": "feature-flags"}
    ):
        return {"flag": event["flag"], "enabled": FLAGS.get(event["flag"], False)}
