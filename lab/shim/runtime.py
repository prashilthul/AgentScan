"""Per-workload runtime: how a real workload talks to the model, to tracing and to "AWS".

Workload code only uses:
    rt.llm(...)                 -> a real model call (Gemini or Anthropic), mirrored per rt.spec.path
    rt.agent_span(name)         -> OTel GenAI `invoke_agent` span (what agent frameworks emit)
    rt.tool_span(name)          -> OTel GenAI `execute_tool` span
    rt.span(name, **attrs)      -> any ordinary span (db query, http call, ...)
    rt.log(msg)                 -> an application log line

Traced workloads get OTel GenAI spans for every model call (`chat <model>` for Gemini, the
SDK's native `anthropic.messages.create` for Anthropic). Untraced workloads emit nothing.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import SpanKind

from lab.shim.mirror import AwsMirror
from lab.shim.providers import AnthropicProvider, GeminiProvider
from lab.spec import Spec


class _Capture(logging.Handler):
    def __init__(self, rt: Runtime):
        super().__init__(logging.INFO)
        self.rt = rt

    def emit(self, record: logging.LogRecord) -> None:
        self.rt.log(record.getMessage())


class Runtime:
    def __init__(
        self,
        spec: Spec,
        state: dict,
        mirror: AwsMirror,
        *,
        span_exporter: SpanExporter | None = None,
        http_client=None,
    ):
        self.spec = spec
        self.state = state
        self.mirror = mirror
        # What the simulated Bedrock route records in CloudTrail. The real inference is served
        # by LAB_PROVIDER; Bedrock itself is part of the fake account.
        self.bedrock_model_id = os.environ.get(
            "SIMULATED_BEDROCK_MODEL_ID", "anthropic.claude-haiku-5-5"
        )
        self.arn = state["arns"][spec.name]
        self.task_id = uuid.uuid4().hex[:32]
        self.request_id: str | None = None
        self.log_group = state["log_groups"].get(spec.name)
        self.log_stream = (
            f"{_today()}/[$LATEST]{uuid.uuid4().hex}"
            if spec.kind == "lambda"
            else f"app/{spec.name}/{self.task_id}"
        )

        self.provider: TracerProvider | None = None
        if spec.traced:
            self.provider = TracerProvider(resource=Resource.create(self._resource_attrs()))
            if span_exporter is None:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                endpoint = os.environ.get("OTLP_ENDPOINT", "http://localhost:4318")
                span_exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
            self.provider.add_span_processor(BatchSpanProcessor(span_exporter))
            self.tracer = self.provider.get_tracer("lab.workload")
        else:
            self.tracer = trace.NoOpTracer()
        self.client = None
        if spec.path != "none":
            if os.environ.get("LAB_PROVIDER", "gemini") == "anthropic":
                self.client = AnthropicProvider(self.provider, http_client=http_client)
            else:
                self.client = GeminiProvider(self.tracer, httpx_client=http_client)

    # ------------------------------------------------------------ identity / resource

    def _resource_attrs(self) -> dict:
        attrs = {
            "service.name": self.spec.name,
            "cloud.provider": "aws",
            "cloud.region": self.state["region"],
            "cloud.account.id": self.state["account_id"],
        }
        if self.spec.kind == "lambda":
            attrs |= {
                "cloud.platform": "aws_lambda",
                "faas.name": self.spec.name,
                "cloud.resource_id": self.arn,
            }
        elif self.spec.kind == "ecs":
            attrs |= {
                "cloud.platform": "aws_ecs",
                "aws.ecs.task.family": self.spec.name,
                "aws.ecs.cluster.arn": self.state["ecs_cluster_arn"],
            }
        return attrs

    @property
    def identity(self) -> dict:
        session = self.spec.name if self.spec.kind == "lambda" else self.task_id
        return self.mirror.assumed_role(self.spec.role_name, session)

    # ------------------------------------------------------------ invocation framing

    @contextmanager
    def invocation(self, name: str) -> Iterator[None]:
        """One request/invocation: Lambda START/END lines + a root SERVER span."""
        self.request_id = str(uuid.uuid4())
        if self.spec.kind == "lambda":
            self.log(f"START RequestId: {self.request_id} Version: $LATEST")
        handlers = []
        if self.spec.path == "direct":
            # The SDKs' HTTP libraries (httpx / httpx2) log every request at INFO - exactly
            # what lands in CloudWatch in prod.
            for name in ("httpx", "httpx2"):
                h = _Capture(self)
                lg = logging.getLogger(name)
                lg.setLevel(logging.INFO)
                lg.addHandler(h)
                handlers.append((lg, h))
        try:
            with self.tracer.start_as_current_span(name, kind=SpanKind.SERVER):
                yield
        finally:
            for lg, h in handlers:
                lg.removeHandler(h)
            if self.spec.kind == "lambda":
                self.log(f"END RequestId: {self.request_id}")
                self.log(f"REPORT RequestId: {self.request_id}\tDuration: 812.40 ms")

    # ------------------------------------------------------------ helpers for workloads

    def llm(self, **request):
        assert self.client is not None, f"{self.spec.name} is not an LLM workload"
        request.setdefault("max_tokens", 2048)
        via_bedrock = self.spec.path in ("bedrock", "agent")
        # One id per Bedrock call, shared by the span, CloudTrail and the invocation log.
        aws_request_id = str(uuid.uuid4()) if via_bedrock else None
        response = self.client.create(
            **request, span_attrs={"aws.request_id": aws_request_id} if via_bedrock else None
        )
        if via_bedrock:
            self.mirror.model_call(
                self.identity, self.bedrock_model_id, request, response, aws_request_id
            )
        return response

    def agent_span(self, agent_name: str):
        return self.tracer.start_as_current_span(
            f"invoke_agent {agent_name}",
            attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": agent_name},
        )

    def tool_span(self, tool_name: str):
        return self.tracer.start_as_current_span(
            f"execute_tool {tool_name}",
            attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool_name},
        )

    def span(self, name: str, **attrs):
        return self.tracer.start_as_current_span(name, attributes=attrs or None)

    def log(self, message: str) -> None:
        if self.log_group:
            self.mirror.app_log(self.log_group, self.log_stream, message)

    def shutdown(self) -> None:
        if self.provider is not None:
            self.provider.force_flush()
            self.provider.shutdown()


def _today() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y/%m/%d")
