"""Discover compute workloads: Lambda, ECS services, SageMaker endpoints, Bedrock agents.

Only names are kept for environment variables. Values can hold secrets; they are read by
the API call (unavoidable) but dropped immediately and never stored or logged.
"""

from __future__ import annotations

from botocore.exceptions import ClientError

from agentscan.aws import Aws
from agentscan.models import AccountSnapshot, Resource


def _paginate(client, op: str, key: str, **kw):
    if client.can_paginate(op):
        for page in client.get_paginator(op).paginate(**kw):
            yield from page.get(key, [])
    else:
        yield from getattr(client, op)(**kw).get(key, [])


def collect_lambda(aws: Aws, region: str, snap: AccountSnapshot) -> None:
    lam = aws.client("lambda", region)
    for fn in _paginate(lam, "list_functions", "Functions"):
        env_names = sorted((fn.get("Environment") or {}).get("Variables", {}).keys())
        try:
            tags = lam.list_tags(Resource=fn["FunctionArn"]).get("Tags", {})
        except ClientError:
            tags = {}
        snap.resources.append(
            Resource(
                arn=fn["FunctionArn"],
                type="lambda.function",
                name=fn["FunctionName"],
                region=region,
                role_arn=fn.get("Role"),
                env_var_names=env_names,
                layers=[layer["Arn"] for layer in fn.get("Layers", [])],
                images=[fn["ImageUri"]] if fn.get("ImageUri") else [],
                log_groups=[
                    (fn.get("LoggingConfig") or {}).get("LogGroup")
                    or f"/aws/lambda/{fn['FunctionName']}"
                ],
                attributes={"runtime": fn.get("Runtime"), "tags": tags},
            )
        )


def collect_ecs(aws: Aws, region: str, snap: AccountSnapshot) -> None:
    ecs = aws.client("ecs", region)
    for cluster_arn in _paginate(ecs, "list_clusters", "clusterArns"):
        service_arns = list(_paginate(ecs, "list_services", "serviceArns", cluster=cluster_arn))
        for i in range(0, len(service_arns), 10):
            described = ecs.describe_services(
                cluster=cluster_arn, services=service_arns[i : i + 10]
            )
            for svc in described.get("services", []):
                td = ecs.describe_task_definition(taskDefinition=svc["taskDefinition"])[
                    "taskDefinition"
                ]
                env_names, secret_refs, images, log_groups = set(), [], [], []
                for c in td.get("containerDefinitions", []):
                    env_names.update(e["name"] for e in c.get("environment", []))
                    for s in c.get("secrets", []):
                        env_names.add(s["name"])
                        secret_refs.append(s["valueFrom"])  # an ARN, not the secret value
                    images.append(c.get("image", ""))
                    lc = c.get("logConfiguration") or {}
                    if lc.get("logDriver") == "awslogs" and lc.get("options", {}).get(
                        "awslogs-group"
                    ):
                        log_groups.append(lc["options"]["awslogs-group"])
                snap.resources.append(
                    Resource(
                        arn=svc["serviceArn"],
                        type="ecs.service",
                        name=svc["serviceName"],
                        region=region,
                        role_arn=td.get("taskRoleArn"),
                        env_var_names=sorted(env_names),
                        secret_refs=secret_refs,
                        images=images,
                        log_groups=log_groups,
                        attributes={"cluster": cluster_arn, "task_family": td.get("family")},
                    )
                )


def collect_sagemaker(aws: Aws, region: str, snap: AccountSnapshot) -> None:
    sm = aws.client("sagemaker", region)
    for ep in _paginate(sm, "list_endpoints", "Endpoints"):
        cfg = sm.describe_endpoint_config(
            EndpointConfigName=sm.describe_endpoint(EndpointName=ep["EndpointName"])[
                "EndpointConfigName"
            ]
        )
        images, env_names, role = [], set(), None
        for variant in cfg.get("ProductionVariants", []):
            model = sm.describe_model(ModelName=variant["ModelName"])
            role = model.get("ExecutionRoleArn")
            containers = model.get("Containers") or [model.get("PrimaryContainer") or {}]
            for c in containers:
                images.append(c.get("Image", ""))
                env_names.update((c.get("Environment") or {}).keys())
        snap.resources.append(
            Resource(
                arn=ep["EndpointArn"],
                type="sagemaker.endpoint",
                name=ep["EndpointName"],
                region=region,
                role_arn=role,
                env_var_names=sorted(env_names),
                images=images,
                log_groups=[f"/aws/sagemaker/Endpoints/{ep['EndpointName']}"],
            )
        )


def collect_bedrock_agents(aws: Aws, region: str, snap: AccountSnapshot) -> None:
    ba = aws.client("bedrock-agent", region)
    for summary in _paginate(ba, "list_agents", "agentSummaries"):
        agent = ba.get_agent(agentId=summary["agentId"])["agent"]
        snap.resources.append(
            Resource(
                arn=agent["agentArn"],
                type="bedrock.agent",
                name=agent["agentName"],
                region=region,
                role_arn=agent.get("agentResourceRoleArn"),
                attributes={
                    "agent_id": agent["agentId"],
                    "foundation_model": agent.get("foundationModel"),
                },
            )
        )
        # Action groups link the agent to the Lambdas that execute its tools.
        try:
            groups = ba.list_agent_action_groups(agentId=agent["agentId"], agentVersion="DRAFT")
            lambdas = []
            for g in groups.get("actionGroupSummaries", []):
                detail = ba.get_agent_action_group(
                    agentId=agent["agentId"],
                    agentVersion="DRAFT",
                    actionGroupId=g["actionGroupId"],
                )["agentActionGroup"]
                fn = (detail.get("actionGroupExecutor") or {}).get("lambda")
                if fn:
                    lambdas.append(fn)
            snap.agent_action_lambdas[agent["agentId"]] = lambdas
        except Exception:  # noqa: BLE001 - moto does not implement action groups
            snap.agent_action_lambdas.setdefault(agent["agentId"], [])


def collect_secret_names(aws: Aws, region: str, snap: AccountSnapshot) -> None:
    sec = aws.client("secretsmanager", region)
    for s in _paginate(sec, "list_secrets", "SecretList"):  # names only, never values
        snap.secret_names.append(s["Name"])


COLLECTORS = {
    "lambda": collect_lambda,
    "ecs": collect_ecs,
    "sagemaker": collect_sagemaker,
    "bedrock-agent": collect_bedrock_agents,
    "secretsmanager": collect_secret_names,
}
