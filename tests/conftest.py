import pytest
from moto import mock_aws


@pytest.fixture
def aws(monkeypatch):
    """In-process moto; never touches a real account or the docker stack."""
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    with mock_aws():
        yield
