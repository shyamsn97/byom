"""Chat, streaming, structured outputs and tool calls through the OpenAI SDK.

    pip install sgbridge openai pydantic
    sgbridge deploy modal --profile qwen3-coder-30b
    python examples/openai_sdk.py [deployment-name]
"""

import json
import sys

from openai import OpenAI
from pydantic import BaseModel

from sgbridge.deploy import Deployment

dep = Deployment.load(sys.argv[1] if len(sys.argv) > 1 else "qwen3-coder-30b")
# A long timeout covers the first request after idle, which waits for the GPU to boot.
client = OpenAI(base_url=f"{dep.url}/v1", api_key=dep.api_key, timeout=900)
model = dep.model

print("== chat")
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Write a one-line Python palindrome check."}],
    max_tokens=200,
)
print(resp.choices[0].message.content)

print("\n== streaming")
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Count from 1 to 10, comma separated."}],
    max_tokens=100,
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()


class Function(BaseModel):
    name: str
    params: list[str]
    returns: str


class ModuleSummary(BaseModel):
    language: str
    functions: list[Function]


print("\n== structured output (JSON schema from a Pydantic model)")
parsed = client.chat.completions.parse(
    model=model,
    messages=[{
        "role": "user",
        "content": "Summarize this code:\n\n"
                   "def add(a: int, b: int) -> int: return a + b\n"
                   "def greet(name: str) -> str: return f'hi {name}'",
    }],
    response_format=ModuleSummary,
    max_tokens=400,
)
print(parsed.choices[0].message.parsed)

print("\n== JSON mode")
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Give three HTTP status codes and meanings as JSON."}],
    response_format={"type": "json_object"},
    max_tokens=300,
)
print(json.loads(resp.choices[0].message.content or "{}"))

print("\n== tool calls")
tools = [{
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": "Run the test suite for a path and return the results.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "verbose": {"type": "boolean"}},
            "required": ["path"],
        },
    },
}]
messages = [{"role": "user", "content": "Run the tests in tests/unit verbosely."}]
resp = client.chat.completions.create(model=model, messages=messages, tools=tools, max_tokens=300)
call = resp.choices[0].message.tool_calls[0]
print(call.function.name, json.loads(call.function.arguments))

# Send the tool result back and let the model finish.
messages += [
    resp.choices[0].message.model_dump(exclude_none=True),
    {"role": "tool", "tool_call_id": call.id, "content": "12 passed, 1 failed: test_parse_empty"},
]
resp = client.chat.completions.create(model=model, messages=messages, tools=tools, max_tokens=300)
print(resp.choices[0].message.content)
