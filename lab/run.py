"""Run every lab workload against its sample events: real code, real LLM calls.

Each run produces what the crawler will later find:
  traces       -> OTLP -> collector -> Jaeger           (instrumented workloads only)
  CloudTrail   -> moto S3 trail bucket                  (Bedrock-path workloads, agent calls)
  model logs   -> moto CloudWatch invocation log group  (Bedrock-path workloads)
  app logs     -> moto CloudWatch, per workload         (all workloads)

Usage: python -m lab.run [--only NAME ...]
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from dotenv import load_dotenv

from lab.shim.mirror import AwsMirror
from lab.shim.runtime import Runtime
from lab.spec import WORKLOADS, Spec


def run_workload(spec: Spec, state: dict, mirror: AwsMirror, **runtime_kw) -> list:
    mod = importlib.import_module(f"lab.workloads.{spec.module}")
    rt = Runtime(spec, state, mirror, **runtime_kw)
    results = []
    try:
        for event in mod.EVENTS:
            with rt.invocation(f"{spec.name} request"):
                results.append(mod.handle(rt, event))
    finally:
        rt.shutdown()
        mirror.flush()
    return results


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="workload names to run")
    args = ap.parse_args()
    state = json.loads(Path("out/lab_state.json").read_text())
    mirror = AwsMirror(state["account_id"], state["region"])
    for spec in WORKLOADS:
        if spec.module is None or (args.only and spec.name not in args.only):
            continue
        print(f"-> {spec.name} ({spec.label}, path={spec.path}, traced={spec.traced})")
        for out in run_workload(spec, state, mirror):
            print(f"   {str(out)[:110]!r}")


if __name__ == "__main__":
    main()
