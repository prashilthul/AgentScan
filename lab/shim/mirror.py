"""Write what AWS *would* have recorded for each real call into the fake account (moto).

  * CloudTrail records   -> S3 bucket of the trail, real delivery key layout, gzipped JSON
  * Bedrock invocation   -> CloudWatch log group configured for model-invocation logging
  * Application logs     -> the workload's own log group (Lambda START/END framing included)

Nothing here is hand-authored per workload: every record is derived from a call that
actually happened in this process.
"""

from __future__ import annotations

import gzip
import json
import uuid
from collections import defaultdict
from datetime import UTC, datetime

import boto3

TRAIL_BUCKET = "acme-cloudtrail-logs"
INVOCATION_LOG_GROUP = "/aws/bedrock/modelinvocations"


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ms(ts: datetime) -> int:
    return int(ts.timestamp() * 1000)


def _block_to_converse(b) -> dict:
    """Anthropic content block -> Bedrock Converse content block."""
    if hasattr(b, "model_dump"):
        b = b.model_dump()
    if isinstance(b, str):
        return {"text": b}
    t = b.get("type")
    if t == "text":
        return {"text": b["text"]}
    if t == "tool_use":
        return {"toolUse": {"toolUseId": b["id"], "name": b["name"], "input": b["input"]}}
    if t == "tool_result":
        content = b.get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        return {"toolResult": {"toolUseId": b["tool_use_id"], "content": [{"text": text}]}}
    return {"text": f"[{t}]"}


def to_converse_messages(messages: list) -> list[dict]:
    out = []
    for m in messages:
        content = m["content"]
        blocks = (
            [{"text": content}]
            if isinstance(content, str)
            else [_block_to_converse(b) for b in content]
        )
        out.append({"role": m["role"], "content": blocks})
    return out


class AwsMirror:
    def __init__(self, account_id: str, region: str):
        self.account_id = account_id
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)
        self.logs = boto3.client("logs", region_name=region)
        self.trail_records: list[dict] = []
        self.invocations: list[tuple[int, str]] = []
        self.app_logs: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)

    # ------------------------------------------------------------ identities

    def assumed_role(self, role_name: str, session: str) -> dict:
        role_arn = f"arn:aws:iam::{self.account_id}:role/{role_name}"
        return {
            "type": "AssumedRole",
            "principalId": f"AROA{uuid.uuid4().hex[:16].upper()}:{session}",
            "arn": f"arn:aws:sts::{self.account_id}:assumed-role/{role_name}/{session}",
            "accountId": self.account_id,
            "sessionContext": {
                "sessionIssuer": {"type": "Role", "arn": role_arn, "userName": role_name},
                "attributes": {
                    "creationDate": _iso(datetime.now(UTC)),
                    "mfaAuthenticated": "false",
                },
            },
        }

    # ------------------------------------------------------------ recorders

    def _trail(
        self,
        identity: dict,
        event_name: str,
        params: dict,
        data_event: bool,
        resources: list[dict] | None = None,
    ) -> None:
        now = datetime.now(UTC)
        rec = {
            "eventVersion": "1.09",
            "userIdentity": identity,
            "eventTime": _iso(now),
            "eventSource": "bedrock.amazonaws.com",
            "eventName": event_name,
            "awsRegion": self.region,
            "sourceIPAddress": "10.0.12.34",
            "userAgent": "Boto3/1.44.11 md/Botocore#1.44.11 ua/2.1 os/linux lang/python#3.13",
            "requestParameters": params,
            "responseElements": None,
            "requestID": str(uuid.uuid4()),
            "eventID": str(uuid.uuid4()),
            "readOnly": True,
            "eventType": "AwsApiCall",
            "managementEvent": not data_event,
            "recipientAccountId": self.account_id,
            "eventCategory": "Data" if data_event else "Management",
        }
        if resources:
            rec["resources"] = resources
        self.trail_records.append(rec)

    def model_call(self, identity: dict, model_id: str, request: dict, response) -> None:
        """One Bedrock Converse call: a CloudTrail event plus an invocation-log entry."""
        self._trail(identity, "Converse", {"modelId": model_id}, data_event=False)
        now = datetime.now(UTC)
        body = {"messages": to_converse_messages(request["messages"])}
        if request.get("system"):
            body["system"] = [{"text": request["system"]}]
        if request.get("tools"):
            body["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "inputSchema": {"json": t["input_schema"]},
                        }
                    }
                    for t in request["tools"]
                ]
            }
        entry = {
            "schemaType": "ModelInvocationLog",
            "schemaVersion": "1.0",
            "timestamp": _iso(now),
            "accountId": self.account_id,
            "identity": {"arn": identity["arn"]},
            "region": self.region,
            "requestId": str(uuid.uuid4()),
            "operation": "Converse",
            "modelId": model_id,
            "input": {
                "inputContentType": "application/json",
                "inputBodyJson": body,
                "inputTokenCount": response.usage.input_tokens,
            },
            "output": {
                "outputContentType": "application/json",
                "outputBodyJson": {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [
                                _block_to_converse(b)
                                for b in response.content
                                if getattr(b, "type", None) in ("text", "tool_use")
                            ],
                        }
                    },
                    "stopReason": response.stop_reason,
                    "usage": {
                        "inputTokens": response.usage.input_tokens,
                        "outputTokens": response.usage.output_tokens,
                    },
                },
                "outputTokenCount": response.usage.output_tokens,
            },
        }
        self.invocations.append((_ms(now), json.dumps(entry)))

    def agent_invoked(self, identity: dict, agent_id: str, agent_arn: str, session_id: str) -> None:
        self._trail(
            identity,
            "InvokeAgent",
            {"agentId": agent_id, "agentAliasId": "TSTALIASID", "sessionId": session_id},
            data_event=True,
            resources=[
                {
                    "type": "AWS::Bedrock::AgentAlias",
                    "ARN": f"{agent_arn.replace(':agent/', ':agent-alias/')}/TSTALIASID",
                }
            ],
        )

    def app_log(self, group: str, stream: str, message: str) -> None:
        self.app_logs[(group, stream)].append((_ms(datetime.now(UTC)), message))

    # ------------------------------------------------------------ flush

    def flush(self) -> None:
        now = datetime.now(UTC)
        if self.trail_records:
            key = (
                f"AWSLogs/{self.account_id}/CloudTrail/{self.region}/{now:%Y/%m/%d}/"
                f"{self.account_id}_CloudTrail_{self.region}_{now:%Y%m%dT%H%MZ}_"
                f"{uuid.uuid4().hex[:16]}.json.gz"
            )
            body = gzip.compress(json.dumps({"Records": self.trail_records}).encode())
            self.s3.put_object(Bucket=TRAIL_BUCKET, Key=key, Body=body)
            self.trail_records = []
        if self.invocations:
            self._put(INVOCATION_LOG_GROUP, "aws/bedrock/modelinvocations", self.invocations)
            self.invocations = []
        for (group, stream), events in self.app_logs.items():
            self._put(group, stream, events)
        self.app_logs.clear()

    def _put(self, group: str, stream: str, events: list[tuple[int, str]]) -> None:
        try:
            self.logs.create_log_stream(logGroupName=group, logStreamName=stream)
        except self.logs.exceptions.ResourceAlreadyExistsException:
            pass
        self.logs.put_log_events(
            logGroupName=group,
            logStreamName=stream,
            logEvents=[
                {"timestamp": t, "message": m} for t, m in sorted(events, key=lambda e: e[0])
            ],
        )
