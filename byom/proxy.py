"""Forwards every request to SGLang unchanged, apart from two fixes for Claude Code requests."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

log = logging.getLogger("byom")

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "host", "content-length",
}
DROP_REQUEST_HEADERS = HOP_BY_HOP | {"accept-encoding"}
DROP_RESPONSE_HEADERS = HOP_BY_HOP | {"content-encoding"}
ATTRIBUTION_PREFIX = "x-anthropic-billing-header:"
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


def fix_request(path: str, body: bytes, drop_thinking: bool) -> bytes:
    """Rewrite an Anthropic Messages body; returns it untouched if nothing needed changing.

    - `thinking`: Claude Code always sends it, and SGLang returns 400 for models served without
      a reasoning parser.
    - The attribution line Claude Code puts first in `system` changes on every request, which
      defeats SGLang's prefix cache.
    """
    if not path.rstrip("/").endswith(("/messages", "/messages/count_tokens")):
        return body
    try:
        data = json.loads(body)
    except ValueError:
        return body
    if not isinstance(data, dict):
        return body
    changed = drop_thinking and data.pop("thinking", None) is not None
    system = data.get("system")
    if isinstance(system, str) and _is_attribution(system):
        data["system"] = _strip_attribution(system)
        changed = True
    elif isinstance(system, list) and system and isinstance(system[0], dict):
        text = system[0].get("text")
        if isinstance(text, str) and _is_attribution(text):
            rest = _strip_attribution(text)
            data["system"] = [{**system[0], "text": rest}, *system[1:]] if rest else system[1:]
            changed = True
    return json.dumps(data, ensure_ascii=False).encode() if changed else body


def _is_attribution(text: str) -> bool:
    return text.lstrip().startswith(ATTRIBUTION_PREFIX)


def _strip_attribution(text: str) -> str:
    return text.lstrip().partition("\n")[2].lstrip("\n")


def create_app(upstream: str, *, drop_thinking: bool = False, hold_s: float = 0,
               transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    """`hold_s` > 0 retries 503s and connection errors that long (a backend cold-starting)."""
    client = httpx.AsyncClient(base_url=upstream.rstrip("/"), transport=transport,
                               timeout=httpx.Timeout(10, read=None))

    async def send(req: httpx.Request) -> httpx.Response | None:
        deadline = time.monotonic() + hold_s
        delay = 1.0
        while True:
            try:
                resp = await client.send(req, stream=True)
                if resp.status_code != 503 or time.monotonic() >= deadline:
                    return resp
                await resp.aclose()
            except httpx.TransportError as e:
                if time.monotonic() >= deadline:
                    log.warning("upstream %s unreachable: %s", upstream, e)
                    return None
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10)

    async def forward(request: Request) -> Response:
        body = await request.body()
        if request.method == "POST":
            body = fix_request(request.url.path, body, drop_thinking)
        headers = {k: v for k, v in request.headers.items() if k not in DROP_REQUEST_HEADERS}
        # Bodies are relayed raw and Content-Encoding is dropped, so they must arrive uncompressed.
        headers["accept-encoding"] = "identity"
        if "authorization" not in headers and "x-api-key" in headers:
            headers["authorization"] = f"Bearer {headers['x-api-key']}"
        req = client.build_request(request.method, request.url.path,
                                   params=request.query_params, headers=headers, content=body)
        resp = await send(req)
        if resp is None:
            return JSONResponse({"error": {"type": "api_error",
                                           "message": f"upstream {upstream} is not reachable"}},
                                status_code=502)
        return StreamingResponse(
            resp.aiter_raw(), status_code=resp.status_code,
            headers={k: v for k, v in resp.headers.items() if k not in DROP_RESPONSE_HEADERS},
            background=BackgroundTask(resp.aclose),
        )

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        yield
        await client.aclose()

    return Starlette(routes=[Route("/{path:path}", forward, methods=METHODS)], lifespan=lifespan)


def serve(upstream: str, *, host: str = "127.0.0.1", port: int = 8080,
          drop_thinking: bool = False, hold_s: float = 0) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(create_app(upstream, drop_thinking=drop_thinking, hold_s=hold_s),
                host=host, port=port)
