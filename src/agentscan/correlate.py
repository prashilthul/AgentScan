"""Group resources into workloads and link each workload to its traces.

A workload is the unit we classify. Labelling resources one by one gives misleading
answers: a Bedrock agent's action-group Lambda is part of the agent, not a separate service.
"""

from __future__ import annotations

from agentscan.models import AccountSnapshot, Resource, TraceStats, Workload


def _matches_trace(res: Resource, stats: TraceStats) -> bool:
    attrs = stats.resource_attrs
    if attrs.get("cloud.resource_id") == res.arn:
        return True
    if res.type == "lambda.function" and attrs.get("faas.name") == res.name:
        return True
    if res.type == "ecs.service" and attrs.get("aws.ecs.task.family") == res.attributes.get(
        "task_family"
    ):
        return True
    return stats.service == res.name


def correlate(snap: AccountSnapshot) -> tuple[list[Workload], list[str]]:
    """Return (workloads, unattributed trace services)."""
    by_arn = {r.arn: r for r in snap.resources}
    absorbed: dict[str, str] = {}  # component ARN -> owning workload ARN

    for res in snap.resources:
        if res.type != "bedrock.agent":
            continue
        for fn_arn in snap.agent_action_lambdas.get(res.attributes.get("agent_id", ""), []):
            if fn_arn in by_arn:
                absorbed[fn_arn] = res.arn

    workloads: dict[str, Workload] = {}
    for res in snap.resources:
        if res.arn in absorbed:
            continue
        workloads[res.arn] = Workload(id=res.arn, name=res.name, primary=res)
    for comp_arn, owner_arn in absorbed.items():
        workloads[owner_arn].components.append(by_arn[comp_arn])
        workloads[owner_arn].notes.append(
            f"{by_arn[comp_arn].name} grouped in as an action-group executor"
        )

    attributed: set[str] = set()
    for wl in workloads.values():
        for res in [wl.primary, *wl.components]:
            for svc, stats in snap.trace_stats.items():
                if _matches_trace(res, stats) and svc not in wl.trace_services:
                    wl.trace_services.append(svc)
                    attributed.add(svc)

    # Roles shared across workloads make role-based evidence ambiguous; flag it.
    role_users: dict[str, list[str]] = {}
    for wl in workloads.values():
        if wl.primary.role_arn:
            role_users.setdefault(wl.primary.role_arn, []).append(wl.name)
    for wl in workloads.values():
        users = role_users.get(wl.primary.role_arn or "", [])
        if len(users) > 1:
            wl.notes.append(
                f"execution role shared with: {', '.join(u for u in users if u != wl.name)}"
            )

    unattributed = sorted(set(snap.trace_stats) - attributed)
    return list(workloads.values()), unattributed
