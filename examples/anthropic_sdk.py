"""Messages, streaming and tool use through the Anthropic SDK (the API Claude Code speaks).

    pip install sgbridge anthropic
    sgbridge deploy modal --profile qwen3-coder-30b
    python examples/anthropic_sdk.py [deployment-name]
"""

import sys

from anthropic import Anthropic

from sgbridge.deploy import Deployment

dep = Deployment.load(sys.argv[1] if len(sys.argv) > 1 else "qwen3-coder-30b")
client = Anthropic(base_url=dep.url, api_key=dep.api_key, timeout=900)
model = dep.model

print("== message")
msg = client.messages.create(
    model=model,
    max_tokens=200,
    messages=[{"role": "user", "content": "What does `git rebase -i` do? One sentence."}],
)
print(msg.content[0].text)

print("\n== streaming")
with client.messages.stream(
    model=model,
    max_tokens=200,
    messages=[{"role": "user", "content": "Name five Unix commands, one per line."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()

print("\n== tool use")
tools = [{
    "name": "read_file",
    "description": "Read a file from the repository.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}]
messages = [{"role": "user", "content": "What's in pyproject.toml?"}]
msg = client.messages.create(model=model, max_tokens=300, tools=tools, messages=messages)
call = next(block for block in msg.content if block.type == "tool_use")
print(call.name, call.input)

messages += [
    {"role": "assistant", "content": msg.content},
    {"role": "user", "content": [{
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": '[project]\nname = "demo"\nversion = "1.2.0"',
    }]},
]
msg = client.messages.create(model=model, max_tokens=300, tools=tools, messages=messages)
print(msg.content[0].text)
