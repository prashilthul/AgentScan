"""Phase 0: an account-wide sweep that assumes nothing about what is running.

Resource Explorer answers "what exists here?" for every resource type in one query. Not every
account has it switched on, so the tagging API is the fallback (it only sees resources that
carry, or once carried, a tag). Either way the result is a plain list of ARNs: the per-service
collectors then read the types agentscan knows how to classify, and the report lists
everything else that can run code as unclassified instead of silently leaving it out.
"""

from __future__ import annotations

import re

from botocore.exceptions import BotoCoreError, ClientError

from agentscan.aws import Aws
from agentscan.models import AccountSnapshot, DiscoveredResource

# Types that can run code (and so could be AI) but have no per-service collector yet.
UNCOVERED_COMPUTE = {
    "ec2.instance",
    "eks.cluster",
    "states.stateMachine",
    "apprunner.service",
    "batch.job-definition",
}


def resource_type(arn: str) -> str:
    """arn:aws:ec2:us-east-1:123:instance/i-1 -> ec2.instance"""
    parts = arn.split(":", 5)
    service, resource = parts[2], parts[5] if len(parts) > 5 else ""
    kind = re.split(r"[:/]", resource, maxsplit=1)
    return f"{service}.{kind[0]}" if len(kind) > 1 else service


def _resource_explorer(aws: Aws, region: str) -> list[str]:
    rex = aws.client("resource-explorer-2", region)
    arns = []
    # shortcut: one Search call returns at most 1,000 resources; page per resource type
    # once accounts get bigger than that.
    for page in rex.get_paginator("search").paginate(QueryString=f"region:{region}"):
        arns += [r["Arn"] for r in page.get("Resources", [])]
    return arns


def _tagging_api(aws: Aws, region: str) -> list[str]:
    tag = aws.client("resourcegroupstaggingapi", region)
    arns = []
    for page in tag.get_paginator("get_resources").paginate():
        arns += [m["ResourceARN"] for m in page.get("ResourceTagMappingList", [])]
    return arns


def collect_discovery(aws: Aws, region: str, snap: AccountSnapshot) -> str:
    try:
        arns, via, note = _resource_explorer(aws, region), "resource-explorer", ""
    except (ClientError, BotoCoreError) as e:
        arns, via = _tagging_api(aws, region), "tagging-api"
        note = f" (Resource Explorer unavailable: {type(e).__name__}; untagged resources missed)"
    seen = {d.arn for d in snap.discovered}
    for arn in arns:
        if arn not in seen:
            seen.add(arn)
            snap.discovered.append(
                DiscoveredResource(arn=arn, type=resource_type(arn), region=region, via=via)
            )
    return f"{len(arns)} resources via {via}{note}"
