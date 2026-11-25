"""Build the fake client account in moto from lab/spec.py.

Writes:
  out/lab_state.json     ARNs / ids the workload runner needs
  out/ground_truth.json  true label per workload, keyed by ARN (input to `agentscan eval`)

Run against a fresh moto (`make down up`); it is not idempotent on purpose.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import boto3
from dotenv import load_dotenv

from lab.shim.mirror import INVOCATION_LOG_GROUP, TRAIL_BUCKET
from lab.spec import WORKLOADS, Spec

TRUST = {
    "lambda": "lambda.amazonaws.com",
    "ecs": "ecs-tasks.amazonaws.com",
    "sagemaker": "sagemaker.amazonaws.com",
    "bedrock-agent": "bedrock.amazonaws.com",
}
CLUSTER = "prod"


def _zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        # The crawler is metadata-only and never downloads code, so the lab ships a stub.
        z.writestr("handler.py", "def handler(event, context):\n    return None\n")
    return buf.getvalue()


def _trust(service: str) -> str:
    return json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}
            ],
        }
    )


def seed(out_dir: Path = Path("out")) -> dict:
    sts = boto3.client("sts")
    region = boto3.session.Session().region_name or "us-east-1"
    account = sts.get_caller_identity()["Account"]
    iam = boto3.client("iam", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    logs = boto3.client("logs", region_name=region)
    ct = boto3.client("cloudtrail", region_name=region)

    # --- account-level audit plumbing
    s3.create_bucket(Bucket=TRAIL_BUCKET)
    ct.create_trail(Name="org-trail", S3BucketName=TRAIL_BUCKET, IsMultiRegionTrail=True)
    ct.start_logging(Name="org-trail")
    logs.create_log_group(logGroupName=INVOCATION_LOG_GROUP)
    boto3.client("bedrock", region_name=region).put_model_invocation_logging_configuration(
        loggingConfig={
            "cloudWatchConfig": {
                "logGroupName": INVOCATION_LOG_GROUP,
                "roleArn": f"arn:aws:iam::{account}:role/bedrock-logging",
            },
            "textDataDeliveryEnabled": True,
        }
    )

    # --- secrets (names only matter to the crawler)
    secret_arns: dict[str, str] = {}
    sm = boto3.client("secretsmanager", region_name=region)
    for spec in WORKLOADS:
        if spec.secret:
            secret_arns[spec.name] = sm.create_secret(
                Name=spec.secret, SecretString="sk-lab-placeholder"
            )["ARN"]

    # --- roles (specs may share one)
    roles: dict[str, list[Spec]] = {}
    for spec in WORKLOADS:
        roles.setdefault(spec.role_name, []).append(spec)
    role_arns: dict[str, str] = {}
    for role_name, specs in roles.items():
        role_arns[role_name] = iam.create_role(
            RoleName=role_name, AssumeRolePolicyDocument=_trust(TRUST[specs[0].kind])
        )["Role"]["Arn"]
        actions = sorted(
            {a for s in specs for a in s.iam_actions if not a.startswith("secretsmanager:")}
        )
        statements = [{"Effect": "Allow", "Action": actions, "Resource": "*"}] if actions else []
        secret_res = [secret_arns[s.name] for s in specs if s.name in secret_arns]
        if secret_res:
            statements.append(
                {
                    "Effect": "Allow",
                    "Action": ["secretsmanager:GetSecretValue"],
                    "Resource": secret_res,
                }
            )
        if statements:
            iam.put_role_policy(
                RoleName=role_name,
                PolicyName="app",
                PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": statements}),
            )
    # Humans reach the Bedrock agent through the console with this role.
    iam.create_role(RoleName="ops-console", AssumeRolePolicyDocument=_trust("ec2.amazonaws.com"))
    iam.put_role_policy(
        RoleName="ops-console",
        PolicyName="invoke",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Action": ["bedrock:InvokeAgent"], "Resource": "*"}
                ],
            }
        ),
    )

    state: dict = {
        "account_id": account,
        "region": region,
        "arns": {},
        "log_groups": {},
        "agent_ids": {},
        "ecs_cluster_arn": None,
    }
    lam = boto3.client("lambda", region_name=region)
    ecs = boto3.client("ecs", region_name=region)
    sage = boto3.client("sagemaker", region_name=region)
    agents = boto3.client("bedrock-agent", region_name=region)
    state["ecs_cluster_arn"] = ecs.create_cluster(clusterName=CLUSTER)["cluster"]["clusterArn"]
    layer_arns: dict[str, str] = {}

    for spec in WORKLOADS:
        role_arn = role_arns[spec.role_name]
        env = dict(spec.env)
        if "API_KEY_SECRET_ARN" in env:
            env["API_KEY_SECRET_ARN"] = secret_arns[spec.name]

        if spec.kind == "lambda":
            for layer in spec.layers:
                if layer not in layer_arns:
                    layer_arns[layer] = lam.publish_layer_version(
                        LayerName=layer,
                        Content={"ZipFile": _zip()},
                        CompatibleRuntimes=["python3.13"],
                    )["LayerVersionArn"]
            fn = lam.create_function(
                FunctionName=spec.name,
                Runtime="python3.13",
                Role=role_arn,
                Handler="handler.handler",
                Code={"ZipFile": _zip()},
                Environment={"Variables": env},
                Layers=[layer_arns[x] for x in spec.layers],
                Tags={"team": "platform", "env": "prod"},
            )
            state["arns"][spec.name] = fn["FunctionArn"]
            group = f"/aws/lambda/{spec.name}"
            logs.create_log_group(logGroupName=group)
            state["log_groups"][spec.name] = group

        elif spec.kind == "ecs":
            group = f"/ecs/{spec.name}"
            logs.create_log_group(logGroupName=group)
            container = {
                "name": "app",
                "image": spec.image,
                "memory": 1024,
                "essential": True,
                "environment": [{"name": k, "value": v} for k, v in env.items()],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": group,
                        "awslogs-region": region,
                        "awslogs-stream-prefix": "app",
                    },
                },
            }
            if spec.name in secret_arns:
                container["secrets"] = [
                    {"name": "GEMINI_API_KEY", "valueFrom": secret_arns[spec.name]}
                ]
            ecs.register_task_definition(
                family=spec.name,
                taskRoleArn=role_arn,
                executionRoleArn=role_arn,
                containerDefinitions=[container],
            )
            svc = ecs.create_service(
                cluster=CLUSTER, serviceName=spec.name, taskDefinition=spec.name, desiredCount=2
            )["service"]
            state["arns"][spec.name] = svc["serviceArn"]
            state["log_groups"][spec.name] = group

        elif spec.kind == "sagemaker":
            sage.create_model(
                ModelName=f"{spec.name}-model",
                ExecutionRoleArn=role_arn,
                PrimaryContainer={"Image": spec.image, "Environment": env},
            )
            sage.create_endpoint_config(
                EndpointConfigName=f"{spec.name}-config",
                ProductionVariants=[
                    {
                        "VariantName": "primary",
                        "ModelName": f"{spec.name}-model",
                        "InitialInstanceCount": 1,
                        "InstanceType": "ml.m5.large",
                    }
                ],
            )
            state["arns"][spec.name] = sage.create_endpoint(
                EndpointName=spec.name, EndpointConfigName=f"{spec.name}-config"
            )["EndpointArn"]

        elif spec.kind == "bedrock-agent":
            agent = agents.create_agent(
                agentName=spec.name,
                foundationModel="anthropic.claude-haiku-5-5",
                agentResourceRoleArn=role_arn,
                instruction="You are an SRE assistant. Investigate alarms and follow runbooks.",
            )["agent"]
            state["arns"][spec.name] = agent["agentArn"]
            state["agent_ids"][spec.name] = agent["agentId"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lab_state.json").write_text(json.dumps(state, indent=2))
    truth = {
        "workloads": [
            {"name": s.name, "arn": state["arns"][s.name], "label": s.label, "why": s.why}
            for s in WORKLOADS
        ]
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    return state


if __name__ == "__main__":
    load_dotenv()
    st = seed()
    print(f"seeded {len(st['arns'])} workloads into account {st['account_id']} ({st['region']})")
    print("wrote out/lab_state.json and out/ground_truth.json")
