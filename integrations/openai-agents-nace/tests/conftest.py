import httpx
import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("a test tried to open an HTTP client")

    monkeypatch.setattr(httpx, "Client", blocked)
    monkeypatch.setattr(httpx, "AsyncClient", blocked)
