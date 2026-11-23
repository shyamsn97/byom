"""End-to-end checks against a real deployment, through the OpenAI and Anthropic SDKs.

    BYOM_LIVE=qwen3-coder-30b uv run pytest tests/test_live.py -v

Skipped unless BYOM_LIVE names a deployment (see `byom list`).
"""

from __future__ import annotations

import json
import os

import anthropic
import openai
import pytest
from pydantic import BaseModel

from byom.deploy import Deployment

NAME = os.environ.get("BYOM_LIVE")
pytestmark = pytest.mark.skipif(not NAME, reason="set BYOM_LIVE=<deployment> to run")

READ_FILE = {
    "name": "read_file",
    "description": "Read a file from the repository.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}


@pytest.fixture(scope="module")
def dep() -> Deployment:
    assert NAME
    return Deployment.load(NAME)


@pytest.fixture(scope="module")
def oai(dep: Deployment) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{dep.url}/v1", api_key=dep.api_key, timeout=900)


@pytest.fixture(scope="module")
def ant(dep: Deployment) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=dep.url, api_key=dep.api_key, timeout=900)


def test_models_lists_served_name(oai: openai.OpenAI, dep: Deployment) -> None:
    assert dep.model in [m.id for m in oai.models.list()]


def test_wrong_key_is_rejected(dep: Deployment) -> None:
    bad = openai.OpenAI(base_url=f"{dep.url}/v1", api_key="sk-wrong", max_retries=0)
    with pytest.raises(openai.AuthenticationError):
        bad.models.list()


def test_openai_chat(oai: openai.OpenAI, dep: Deployment) -> None:
    resp = oai.chat.completions.create(
        model=dep.model, max_tokens=20, temperature=0,
        messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
    )
    assert "pong" in (resp.choices[0].message.content or "").lower()
    assert resp.usage and resp.usage.completion_tokens > 0


def test_openai_stream(oai: openai.OpenAI, dep: Deployment) -> None:
    stream = oai.chat.completions.create(
        model=dep.model, max_tokens=60, temperature=0, stream=True,
        messages=[{"role": "user", "content": "Count from 1 to 10, comma separated."}],
    )
    pieces = [c.choices[0].delta.content for c in stream
              if c.choices and c.choices[0].delta.content]
    assert len(pieces) > 3
    assert "10" in "".join(pieces)


class Person(BaseModel):
    name: str
    age: int
    languages: list[str]


def test_openai_structured_output(oai: openai.OpenAI, dep: Deployment) -> None:
    resp = oai.chat.completions.parse(
        model=dep.model, max_tokens=200, temperature=0, response_format=Person,
        messages=[{"role": "user",
                   "content": "Ada is 36 and writes Python and Rust. Extract her details."}],
    )
    person = resp.choices[0].message.parsed
    assert person is not None
    assert person.name == "Ada" and person.age == 36
    assert {lang.lower() for lang in person.languages} == {"python", "rust"}


def test_openai_json_mode(oai: openai.OpenAI, dep: Deployment) -> None:
    resp = oai.chat.completions.create(
        model=dep.model, max_tokens=200, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": "Give a JSON object mapping 'a' to 1 and 'b' to 2."}],
    )
    assert json.loads(resp.choices[0].message.content or "") == {"a": 1, "b": 2}


def test_openai_tool_call(oai: openai.OpenAI, dep: Deployment) -> None:
    resp = oai.chat.completions.create(
        model=dep.model, max_tokens=200, temperature=0,
        tools=[{"type": "function", "function": READ_FILE}],
        messages=[{"role": "user", "content": "Show me the contents of setup.py."}],
    )
    calls = resp.choices[0].message.tool_calls
    assert calls and calls[0].function.name == "read_file"
    assert json.loads(calls[0].function.arguments)["path"].endswith("setup.py")


def test_anthropic_message(ant: anthropic.Anthropic, dep: Deployment) -> None:
    msg = ant.messages.create(
        model=dep.model, max_tokens=20,
        messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
    )
    assert msg.content[0].type == "text"
    assert "pong" in msg.content[0].text.lower()


def test_anthropic_stream(ant: anthropic.Anthropic, dep: Deployment) -> None:
    with ant.messages.stream(
        model=dep.model, max_tokens=60,
        messages=[{"role": "user", "content": "Count from 1 to 10, comma separated."}],
    ) as stream:
        pieces = list(stream.text_stream)
    assert len(pieces) > 3
    assert "10" in "".join(pieces)


def test_anthropic_tool_use(ant: anthropic.Anthropic, dep: Deployment) -> None:
    tool = {"name": READ_FILE["name"], "description": READ_FILE["description"],
            "input_schema": READ_FILE["parameters"]}
    msg = ant.messages.create(
        model=dep.model, max_tokens=200, tools=[tool],  # type: ignore[list-item]
        messages=[{"role": "user", "content": "Show me the contents of setup.py."}],
    )
    calls = [b for b in msg.content if b.type == "tool_use"]
    assert msg.stop_reason == "tool_use"
    assert calls and calls[0].name == "read_file"
    assert str(calls[0].input["path"]).endswith("setup.py")  # type: ignore[index]


def test_anthropic_thinking_is_accepted(ant: anthropic.Anthropic, dep: Deployment) -> None:
    """Claude Code always sends `thinking`; profiles with drop_thinking must not 400 on it."""
    msg = ant.messages.create(
        model=dep.model, max_tokens=2048, thinking={"type": "enabled", "budget_tokens": 1024},
        messages=[{"role": "user", "content": "Say hi."}],
    )
    assert any(b.type == "text" for b in msg.content)
