"""The simulated client account: every workload, how it is deployed, and its true label.

This file is the ground truth. The crawler never reads it; `lab.seed` turns it into fake AWS
resources (moto) and writes out/ground_truth.json keyed by ARN for `agentscan eval`.

Each workload is real code in lab/workloads/ that makes real LLM calls (or none). The spec
only decides how it LOOKS to AWS: resource type, IAM, env var names, layers, whether it is
instrumented with OpenTelemetry, and whether it reaches the model via Bedrock or directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["lambda", "ecs", "sagemaker", "agentcore"]
Label = Literal["agent", "llm", "service"]
# bedrock:  the call is mirrored into CloudTrail + Bedrock invocation logs (as AWS would)
# direct:   the workload calls the provider API (Gemini / Anthropic) itself; AWS sees nothing but app logs
# agent:    a Bedrock AgentCore runtime: CloudTrail records InvokeAgentRuntime calls made to it,
#           and its own model calls go through Bedrock like the bedrock path
# none:     no LLM at all
Path = Literal["bedrock", "direct", "agent", "none"]

BEDROCK_INVOKE = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
LOGS = ["logs:CreateLogStream", "logs:PutLogEvents"]


@dataclass
class Spec:
    name: str
    kind: Kind
    label: Label
    why: str
    path: Path
    traced: bool
    module: str | None = None  # lab.workloads.<module>
    iam_actions: list[str] = field(default_factory=list)
    secret: str | None = None  # Secrets Manager name the workload reads
    env: dict[str, str] = field(default_factory=dict)
    layers: list[str] = field(default_factory=list)
    image: str | None = None
    role: str | None = None  # shared role name; default "<name>-role"

    @property
    def role_name(self) -> str:
        return self.role or f"{self.name}-role"


WORKLOADS: list[Spec] = [
    # ------------------------------------------------------------------ agents
    Spec(
        name="support-agent",
        kind="lambda",
        label="agent",
        why="Tool loop: looks up orders and refund policy, may issue a refund, then answers.",
        path="bedrock",
        traced=True,
        module="support_agent",
        iam_actions=BEDROCK_INVOKE + ["dynamodb:GetItem", "dynamodb:UpdateItem"] + LOGS,
        env={
            "BEDROCK_MODEL_ID": "anthropic.claude-haiku-5-5",
            "AGENT_MAX_STEPS": "6",
            "ORDERS_TABLE": "orders",
        },
        layers=["anthropic-sdk-2-14"],
    ),
    Spec(
        name="research-crew",
        kind="ecs",
        label="agent",
        why="Multi-agent: planner -> researcher (search/fetch tools, loops) -> writer.",
        path="direct",
        traced=True,
        module="research_crew",
        iam_actions=["secretsmanager:GetSecretValue", "s3:GetObject"] + LOGS,
        secret="prod/research-crew/gemini-api-key",
        env={"KB_BUCKET": "acme-kb"},
        image="123456789012.dkr.ecr.us-east-1.amazonaws.com/research-crew:1.3.0",
    ),
    Spec(
        name="ops-copilot",
        kind="agentcore",
        label="agent",
        why="Bedrock AgentCore runtime; checks alarms, reads runbooks, restarts services. "
        "Invoked by on-call engineers. NOT instrumented: CloudTrail and invocation logs only.",
        path="agent",
        traced=False,
        module="ops_copilot",
        iam_actions=BEDROCK_INVOKE + ["ecs:UpdateService", "cloudwatch:DescribeAlarms"],
        env={"MODEL_ID": "anthropic.claude-haiku-5-5", "AGENT_MAX_STEPS": "8"},
        image="123456789012.dkr.ecr.us-east-1.amazonaws.com/ops-copilot:2.0.0",
    ),
    Spec(
        name="invoice-agent",
        kind="lambda",
        label="agent",
        why="Tool loop (vendor lookup, PO match, approve/flag). NOT instrumented: only "
        "Bedrock invocation logs reveal the loop.",
        path="bedrock",
        traced=False,
        module="invoice_agent",
        iam_actions=BEDROCK_INVOKE + LOGS,
        env={"MODEL_ID": "anthropic.claude-haiku-5-5"},
    ),
    # ------------------------------------------------------------------ single-call LLM
    Spec(
        name="ticket-summarizer",
        kind="lambda",
        label="llm",
        why="One summarisation call per ticket. Shares an execution role with orders-api.",
        path="bedrock",
        traced=True,
        module="ticket_summarizer",
        iam_actions=BEDROCK_INVOKE + ["dynamodb:GetItem", "dynamodb:PutItem"] + LOGS,
        env={"BEDROCK_MODEL_ID": "anthropic.claude-haiku-5-5"},
        role="lambda-shared-exec",
    ),
    Spec(
        name="docs-qa",
        kind="ecs",
        label="llm",
        why="RAG: code retrieves documents, then ONE model call. Retrieval is not a tool "
        "the model chooses.",
        path="bedrock",
        traced=True,
        module="docs_qa",
        iam_actions=BEDROCK_INVOKE + ["s3:GetObject"] + LOGS,
        env={"LLM_MODEL": "anthropic.claude-haiku-5-5", "INDEX_BUCKET": "acme-docs-index"},
        image="123456789012.dkr.ecr.us-east-1.amazonaws.com/docs-qa:2.0.1",
    ),
    Spec(
        name="review-sentiment",
        kind="lambda",
        label="llm",
        why="One call per review, straight to the provider API. NOT instrumented, not via "
        "Bedrock: only app logs and IAM secret access give it away.",
        path="direct",
        traced=False,
        module="review_sentiment",
        iam_actions=["secretsmanager:GetSecretValue"] + LOGS,
        secret="prod/review-sentiment/gemini-api-key",
        env={"API_KEY_SECRET_ARN": "(set at deploy)"},
    ),
    Spec(
        name="invoice-extractor",
        kind="lambda",
        label="llm",
        why="Offers a strict tool schema purely to get JSON back; the tool result is never "
        "fed back. Looks agent-ish, is a single call.",
        path="bedrock",
        traced=True,
        module="invoice_extractor",
        iam_actions=BEDROCK_INVOKE + ["s3:GetObject"] + LOGS,
        env={"BEDROCK_MODEL_ID": "anthropic.claude-haiku-5-5"},
        layers=["anthropic-sdk-2-14"],
    ),
    # ------------------------------------------------------------------ plain services
    Spec(
        name="orders-api",
        kind="lambda",
        label="service",
        why="CRUD API. Shares a role that is allowed to call Bedrock, never does.",
        path="none",
        traced=True,
        module="orders_api",
        role="lambda-shared-exec",
        env={"ORDERS_TABLE": "orders"},
    ),
    Spec(
        name="nightly-report",
        kind="lambda",
        label="service",
        why="Cron job with a leftover bedrock:* permission it never uses. Not instrumented.",
        path="none",
        traced=False,
        module="nightly_report",
        iam_actions=["bedrock:*", "s3:PutObject", "dynamodb:Scan"] + LOGS,
        env={"REPORT_BUCKET": "acme-reports"},
    ),
    Spec(
        name="ai-helper-svc",
        kind="ecs",
        label="service",
        why="Misleading name: a feature-flag lookup service with no AI at all.",
        path="none",
        traced=True,
        module="ai_helper_svc",
        iam_actions=["dynamodb:GetItem"] + LOGS,
        env={"FLAGS_TABLE": "feature-flags"},
        image="123456789012.dkr.ecr.us-east-1.amazonaws.com/ai-helper-svc:0.9.2",
    ),
    Spec(
        name="fraud-score",
        kind="sagemaker",
        label="service",
        why="Classical ML (XGBoost) endpoint - not an LLM.",
        path="none",
        traced=False,
        module=None,
        image="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1",
        env={"SAGEMAKER_PROGRAM": "inference.py"},
    ),
]

BY_NAME = {s.name: s for s in WORKLOADS}
