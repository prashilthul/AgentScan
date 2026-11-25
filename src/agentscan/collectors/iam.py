"""Read the IAM policies attached to each workload's execution role.

IAM says what a workload is *allowed* to do. That is declared evidence only: plenty of
roles carry bedrock:* and never call Bedrock. CloudTrail is what proves usage.
"""

from __future__ import annotations

import json
from urllib.parse import unquote

from agentscan.aws import Aws
from agentscan.models import AccountSnapshot, RolePolicy


def _as_list(v) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _normalise(doc) -> list[dict]:
    if isinstance(doc, str):
        doc = json.loads(unquote(doc))
    out = []
    for st in _as_list(doc.get("Statement")):
        out.append(
            {
                "effect": st.get("Effect", "Allow"),
                "actions": [a.lower() for a in _as_list(st.get("Action"))],
                "resources": _as_list(st.get("Resource")),
            }
        )
    return out


def collect_iam(aws: Aws, region: str, snap: AccountSnapshot) -> None:
    iam = aws.client("iam", region)
    role_arns = {r.role_arn for r in snap.resources if r.role_arn}
    for role_arn in sorted(role_arns):
        name = role_arn.rsplit("/", 1)[-1]
        statements: list[dict] = []
        for p in iam.list_role_policies(RoleName=name).get("PolicyNames", []):
            doc = iam.get_role_policy(RoleName=name, PolicyName=p)["PolicyDocument"]
            statements += _normalise(doc)
        for p in iam.list_attached_role_policies(RoleName=name).get("AttachedPolicies", []):
            pol = iam.get_policy(PolicyArn=p["PolicyArn"])["Policy"]
            ver = iam.get_policy_version(
                PolicyArn=p["PolicyArn"], VersionId=pol["DefaultVersionId"]
            )["PolicyVersion"]["Document"]
            statements += _normalise(ver)
        snap.role_policies[role_arn] = RolePolicy(role_arn=role_arn, statements=statements)
