"""boto3 session helper.

boto3 honours AWS_ENDPOINT_URL natively, so pointing the crawler at moto (local) or at a
real account is configuration only - no code path differs.
"""

from __future__ import annotations

import boto3
from botocore.config import Config

_CFG = Config(retries={"max_attempts": 5, "mode": "adaptive"}, user_agent_extra="agentscan")


class Aws:
    def __init__(self, profile: str | None = None):
        self.session = boto3.Session(profile_name=profile)
        self._clients: dict[tuple[str, str], object] = {}

    def client(self, service: str, region: str):
        key = (service, region)
        if key not in self._clients:
            self._clients[key] = self.session.client(service, region_name=region, config=_CFG)
        return self._clients[key]

    def account_id(self, region: str) -> str:
        return self.client("sts", region).get_caller_identity()["Account"]
