from __future__ import annotations

import gzip
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from sgbridge.proxy import ATTRIBUTION_PREFIX, create_app, fix_request

ATTR = f"{ATTRIBUTION_PREFIX} cc_version=2.1.3; cch=abc;"


class FakeSGLang:
    """Records requests; replies 503 for the first `cold` calls, then streams `chunks`."""

    def __init__(self, cold: int = 0, chunks: tuple[bytes, ...] = (b"data: 1\n\n", b"data: 2\n\n")):
        self.cold = cold
        self.chunks = chunks
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.cold > 0:
            self.cold -= 1
            return httpx.Response(503, content=_stream((b"starting",)))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=_stream(self.chunks))


async def _stream(chunks: tuple[bytes, ...]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def client_for(fake: FakeSGLang, **kwargs: object) -> httpx.AsyncClient:
    app = create_app("http://sglang:30000", transport=httpx.MockTransport(fake), **kwargs)  # type: ignore[arg-type]
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("sgbridge.proxy.asyncio.sleep", no_sleep)


async def test_passes_bytes_through_untouched() -> None:
    fake = FakeSGLang()
    body = b'{"model": "qwen3-coder",   "messages": [], "stream": true}'
    async with client_for(fake) as c:
        resp = await c.post("/v1/messages?beta=true", content=body,
                            headers={"authorization": "Bearer k", "x-custom": "1"})
    assert resp.status_code == 200
    assert resp.content == b"data: 1\n\ndata: 2\n\n"
    sent = fake.requests[0]
    assert sent.content == body
    assert str(sent.url) == "http://sglang:30000/v1/messages?beta=true"
    assert sent.headers["authorization"] == "Bearer k"
    assert sent.headers["x-custom"] == "1"


async def test_x_api_key_becomes_bearer() -> None:
    fake = FakeSGLang()
    async with client_for(fake) as c:
        await c.get("/v1/models", headers={"x-api-key": "sk-1"})
    assert fake.requests[0].headers["authorization"] == "Bearer sk-1"


async def test_claude_code_fixes() -> None:
    fake = FakeSGLang()
    body = {"model": "m", "thinking": {"type": "adaptive"}, "messages": [],
            "system": [{"type": "text", "text": ATTR}, {"type": "text", "text": "You are CC."}]}
    async with client_for(fake, drop_thinking=True) as c:
        await c.post("/v1/messages", json=body)
    sent = json.loads(fake.requests[0].content)
    assert "thinking" not in sent
    assert sent["system"] == [{"type": "text", "text": "You are CC."}]


def test_fix_request_cases() -> None:
    keep = json.dumps({"thinking": {"type": "enabled"}, "system": "plain"}).encode()
    assert fix_request("/v1/messages", keep, drop_thinking=False) is keep
    assert fix_request("/v1/chat/completions", keep, drop_thinking=True) is keep
    assert fix_request("/v1/messages", b"not json", drop_thinking=True) == b"not json"
    stripped = fix_request("/v1/messages", json.dumps({"system": ATTR + "\nHi"}).encode(), False)
    assert json.loads(stripped) == {"system": "Hi"}


async def test_responses_are_not_compressed() -> None:
    body = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()

    def gzips_when_allowed(request: httpx.Request) -> httpx.Response:
        if "gzip" in request.headers.get("accept-encoding", ""):
            return httpx.Response(200, headers={"content-encoding": "gzip"},
                                  content=_stream((gzip.compress(body),)))
        return httpx.Response(200, content=_stream((body,)))

    app = create_app("http://sglang:30000", transport=httpx.MockTransport(gzips_when_allowed))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://proxy") as c:
        resp = await c.post("/v1/chat/completions", json={}, headers={"accept-encoding": "gzip"})
    assert resp.json() == {"choices": [{"message": {"content": "hi"}}]}


async def test_holds_while_backend_starts() -> None:
    fake = FakeSGLang(cold=3)
    async with client_for(fake, hold_s=60) as c:
        resp = await c.post("/v1/messages", json={"messages": []})
    assert resp.status_code == 200
    assert len(fake.requests) == 4


async def test_without_hold_503_is_returned() -> None:
    async with client_for(FakeSGLang(cold=1)) as c:
        resp = await c.get("/health")
    assert resp.status_code == 503
    assert resp.text == "starting"


async def test_unreachable_upstream_is_502() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    app = create_app("http://sglang:30000", transport=httpx.MockTransport(refuse))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://proxy") as c:
        resp = await c.get("/v1/models")
    assert resp.status_code == 502
    assert "not reachable" in resp.json()["error"]["message"]
