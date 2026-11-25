"""Multi-agent research service (ECS, direct to api.anthropic.com).

planner (1 call) -> researcher (tool loop over a knowledge base) -> writer (1 call)
"""

from lab.workloads._loop import run_agent, tool

KB = {
    "kb-1": "Our EU region launched in March; latency dropped 40% for EU customers.",
    "kb-2": "Support tickets about invoices doubled after the pricing change in Q2.",
    "kb-3": "The mobile app's crash rate fell to 0.2% after the 4.1 release.",
    "kb-4": "Enterprise churn is concentrated in accounts with no admin training.",
}

TOOLS = [
    tool(
        "search_kb",
        "Keyword search over internal notes; returns matching doc ids.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    tool("fetch_doc", "Fetch one note by id.", {"doc_id": {"type": "string"}}, ["doc_id"]),
]


def _search(query: str):
    words = {w.lower() for w in query.split()}
    return [
        k for k, v in KB.items() if words & {w.lower().strip(".,%") for w in v.split()}
    ] or list(KB)


IMPLS = {"search_kb": _search, "fetch_doc": lambda doc_id: KB.get(doc_id, "not found")}

EVENTS = [
    {"question": "Why are enterprise customers churning?"},
    {"question": "What changed for EU customers this year?"},
    {"question": "Is invoice-related support load going up?"},
]


def handle(rt, event):
    q = event["question"]
    with rt.agent_span("planner"):
        plan = rt.llm(
            system="Turn the question into 2 short search queries, one per line. No prose.",
            messages=[{"role": "user", "content": q}],
            max_tokens=200,
        )
    queries = "".join(b.text for b in plan.content if b.type == "text")
    notes = run_agent(
        rt,
        "researcher",
        "Research the question with the tools. Fetch the relevant notes, then list the facts.",
        f"Question: {q}\nSuggested queries:\n{queries}",
        TOOLS,
        IMPLS,
        max_steps=5,
    )
    with rt.agent_span("writer"):
        final = rt.llm(
            system="Write a 3-sentence answer for an executive from the research notes.",
            messages=[{"role": "user", "content": f"Question: {q}\nNotes:\n{notes}"}],
            max_tokens=400,
        )
    return "".join(b.text for b in final.content if b.type == "text")
