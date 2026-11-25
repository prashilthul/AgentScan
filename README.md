# agentscan

agentscan is a read-only crawler for an AWS account. It labels every workload as an agent, a
single-call LLM app or a plain service. Each label has a confidence score and the evidence
behind it.

## What runs where

`make up` starts three containers. The lab workloads and the crawler run on your machine.

| Service | Port | Stands in for | What you can do there |
|---|---|---|---|
| moto server | 5000 | the client's AWS account | boto3 talks to it through `AWS_ENDPOINT_URL` |
| OTel collector | 4318 | the AWS Distro for OpenTelemetry collector | receives spans over OTLP/HTTP and forwards them to Jaeger |
| Jaeger | 16686 | AWS X-Ray | open http://localhost:16686 to browse traces |

## Commands

You need Docker, [uv](https://docs.astral.sh/uv/) and a free Gemini key from
https://aistudio.google.com/apikey.

```bash
cp .env.example .env          # add GEMINI_API_KEY
make up                       # start the moto server, the OTel collector and Jaeger
make seed                     # create the fake client account; writes out/ground_truth.json
make run                      # run the 12 workloads with real model calls, about 10 minutes
make scan                     # crawl and classify; writes out/report.json
make eval                     # accuracy and confusion matrix against the ground truth
uv run agentscan explain out/report.json invoice-agent   # every piece of evidence for one label
make test                     # offline test suite, no Docker or key needed
```

Useful flags on `agentscan scan`: `--disable traces` turns one source off to show what
it adds, `--no-judge` skips the LLM review, and `--lookback-hours` and `--region` set the window.
To scan a real account, unset `AWS_ENDPOINT_URL` and pass `--profile <read-only-role>`.

## OpenTelemetry data

Half the lab's agents are instrumented and half are not, as the brief asked.

- `support-agent` and `research-crew` send spans over OTLP/HTTP to the collector on port 4318.
  The collector batches them and exports them to Jaeger. The config is
  `infra/otel-collector.yaml`.
- `ops-copilot` and `invoice-agent` send nothing. `ops-copilot` is a Bedrock agent, so the
  crawler sees it in the inventory and in CloudTrail's `InvokeAgent` records. `invoice-agent`
  shows its tool loop only in the Bedrock invocation logs.

The spans follow the OpenTelemetry GenAI conventions. A model call is a `chat <model>` span, a
tool run is `execute_tool <tool>` and an agent step is `invoke_agent <name>`. Each span carries
resource attributes: `service.name`, `cloud.resource_id` for Lambda, `faas.name` and
`aws.ecs.task.family`.

The crawler reads traces from Jaeger's query API. It counts model calls, tool calls and agent
spans per trace and checks their order. It never reads prompt or completion text.

## How a workload gets its label

| Label | What it means |
|---|---|
| service | No evidence that it calls a model. Classical ML, such as an XGBoost model, counts here. |
| llm | It calls a model once per request. The code builds the prompt and uses the answer. No tool result goes back to the model. |
| agent | Inside one request, the model asks for a tool, the tool result goes back to the model, and the model is called again. Several agents handing work to each other also count. |

The crawler answers two questions for every workload.

The first question is whether it uses a model. These records say yes:
- CloudTrail shows `InvokeModel` or `Converse` calls made by the workload's role.
- The Bedrock invocation logs show requests from the workload's role.
- The app logs show HTTP calls to a model provider, such as `generativelanguage.googleapis.com`.
- A trace holds a model call span.

The second question is whether the model loops with tools. These records say yes:
- A trace shows a model call that starts after a tool call ended.
- A Bedrock invocation log request contains a tool result from the previous call.
- A trace holds spans from several different agents.
- The app logs show several model calls inside one Lambda invocation.

IAM permissions, environment variable names, SDK packages and secret names are hints only. They
show what a workload could do, so a label built on them alone stays at 0.75 confidence or below.

## How it works

1. Inventory. The crawler reads the configuration of every Lambda, ECS service, SageMaker
   endpoint and Bedrock agent. It keeps environment variable names and drops
   their values.
2. Grouping. Resources become workloads. A Bedrock agent and the Lambdas that run its tools are
   one workload.
3. Evidence. Each source turns into evidence with a strength. Configuration and IAM
   are declared evidence and show what a workload could do. CloudTrail, invocation logs and app
   logs are observed evidence and show that it called a model. Traces are behavioral evidence
   and show how it called the model.
4. Scoring. Two questions are scored separately: does the workload use a model, and does it loop
   with tools. Each piece of evidence adds weight to one or both. The weights live in one table
   in `src/agentscan/signals.py`.
5. Confidence cap. Confidence never goes above what the strongest evidence allows. The caps are
   0.75 for declared, 0.90 for observed and 0.97 for behavioral evidence.
6. Review. When confidence is below 0.7, an LLM judge reads the evidence list and gives its
   verdict. The scan discards the verdict if it cites no real evidence id. The same caps apply.
7. Report. The scan writes `out/report.json`. Every source reports whether it worked, so
   missing access shows up as missing access and not as "no AI found".

## Which workload owns a record

The crawler links every record to a workload by an identifier that the record already carries.
It never uses names or tags as evidence.

| Record | How the crawler finds the owner |
|---|---|
| Trace span | `cloud.resource_id` matches the Lambda ARN, `faas.name` matches the function, or `aws.ecs.task.family` matches the task family. `service.name` is the fallback. |
| CloudTrail record | The caller is an assumed role. If one workload uses that role, it owns the record. If several share the role, the session name must match a Lambda name. Otherwise the record stays unattributed. |
| Bedrock invocation log | The same role and session rule as CloudTrail. |
| App log line | The log group belongs to one workload, such as `/aws/lambda/<name>` or `/ecs/<name>`. |
| Bedrock agent call | The agent id in CloudTrail's `InvokeAgent` record. |

A role that several workloads share counts at half weight as evidence, and the report flags it.
"Who" in this sense is the AWS identity that made the call. The crawler does not know which team
owns a workload.

## Decisions and trade-offs

- Evidence strength caps confidence. A workload with `bedrock:*` in its role and no recorded calls
  stays a service. I chose under-claiming over a confident label an auditor can't check.
- Two questions instead of three classes. An agent is a kind of AI workload, so the labels are not
  independent. "CloudTrail shows model calls" answers the first question and says nothing about
  the second.
- The loop is the agent test. A model call that starts after a tool finished, in the same
  request, means the tool result went back to the model. A tool schema alone does not make an
  agent.
- Rules decide and the LLM reviews. Every weight is readable in one file. The judge sees evidence,
  not raw logs, and must cite it. A black box would be hard to defend in an audit.
- Names are never evidence. `ai-helper-svc` in the lab is a feature-flag lookup.
- The lab uses real model calls. Only AWS is fake, and the fake AWS records come from calls that
  happened. This costs free-tier quota and run time, and it makes the accuracy numbers real.

## Edge cases

| Case | Label | How agentscan decides |
|---|---|---|
| Multi-turn chat | llm | It counts model calls per request, not per conversation. |
| Chat that calls tools inside one turn | agent | A tool result goes back to the model. |
| Tool schema used only to get JSON | llm | Tools are offered, but no result is ever sent back. The lab's `invoice-extractor` does this. |
| RAG | llm | The code chooses to retrieve, and the model is called once. The lab's `docs-qa` does this. |
| Several agents passing work | agent | One trace holds several agent spans. The lab's `research-crew` does this. |
| Untraced agent on ECS that calls a provider directly | agent, but seen as llm | ECS logs don't separate requests, so only traces can tell. |
| Bedrock permission that is never used | service | CloudTrail shows no calls. The lab's `nightly-report` does this. |
| Classical ML on SageMaker | service | The question is about LLMs. The serving image shows XGBoost. |

## Assumptions

- The client grants a read-only role per data source, and source code is out of scope.
- A Lambda's CloudTrail session name is the function name, which is the AWS default.
- Teams that trace use the OpenTelemetry GenAI conventions.
- Each workload ran at least once during the lookback window, which is 24 hours by default.
- Every workload has its own log group, such as `/aws/lambda/<name>` or `/ecs/<name>`.

## Results

- The offline suite runs without Docker. It gets 12/12 with every source on and
  11/12 without traces. The miss is `research-crew`, which calls Gemini directly from ECS.

## Layout

```
src/agentscan/   the crawler: collectors, correlate, signals, scoring, judge, cli
lab/             the fake client account: spec.py holds the ground truth
tests/           offline end-to-end and unit tests
infra/           the OTel collector config
```

The read-only IAM permissions for each source are in `src/agentscan/access.py`.
