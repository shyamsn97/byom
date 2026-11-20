# Using a deployment

A deployment is one URL that serves both APIs SGLang speaks:

| API | Base URL | Used by |
|---|---|---|
| Anthropic Messages | `<url>` | Claude Code, the `anthropic` SDK |
| OpenAI Chat Completions | `<url>/v1` | the `openai` SDK, Cursor, most other tools |

Both take the `sk-sgb-...` key and the profile's served model name (`qwen3-coder` for the bundled
profile). `sgbridge env NAME` prints all three; in Python, `Deployment.load(NAME)` has them as
`.url`, `.api_key` and `.model`.

The first request after the GPU has been idle waits for it to boot (2-4 minutes on Modal) instead
of failing, so give clients a long timeout (`timeout=900` below).

## Claude Code

```bash
eval "$(sgbridge env qwen3-coder-30b)"
claude
```

[`examples/claude_code_qwen.sh`](../examples/claude_code_qwen.sh) deploys and has Claude Code
write and test a small program headlessly.

## OpenAI SDK

```python
from openai import OpenAI
from sgbridge.deploy import Deployment

dep = Deployment.load("qwen3-coder-30b")
client = OpenAI(base_url=f"{dep.url}/v1", api_key=dep.api_key, timeout=900)
```

Chat:

```python
resp = client.chat.completions.create(
    model=dep.model,
    messages=[{"role": "user", "content": "Write a one-line palindrome check."}],
)
print(resp.choices[0].message.content)
```

Streaming:

```python
stream = client.chat.completions.create(model=dep.model, messages=[...], stream=True)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Structured outputs. SGLang enforces the JSON schema while decoding, so the output always parses:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

resp = client.chat.completions.parse(
    model=dep.model,
    messages=[{"role": "user", "content": "Ada is 36. Extract her details."}],
    response_format=Person,
)
person = resp.choices[0].message.parsed  # Person(name='Ada', age=36)
```

`response_format={"type": "json_object"}` works too, for JSON without a fixed schema.

Tool calls:

```python
tools = [{"type": "function", "function": {
    "name": "read_file",
    "description": "Read a file from the repository.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]},
}}]
resp = client.chat.completions.create(model=dep.model, messages=[...], tools=tools)
call = resp.choices[0].message.tool_calls[0]  # call.function.name, call.function.arguments
```

Send the result back as a `{"role": "tool", "tool_call_id": call.id, "content": ...}` message.
Full runnable version: [`examples/openai_sdk.py`](../examples/openai_sdk.py).

## Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(base_url=dep.url, api_key=dep.api_key, timeout=900)

msg = client.messages.create(model=dep.model, max_tokens=500,
                             messages=[{"role": "user", "content": "Hi"}])

with client.messages.stream(model=dep.model, max_tokens=500, messages=[...]) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

Tool use takes `tools=[{"name": ..., "description": ..., "input_schema": {...}}]` and returns
`tool_use` blocks, as with Claude. Full version: [`examples/anthropic_sdk.py`](../examples/anthropic_sdk.py).

## curl

```bash
eval "$(sgbridge env qwen3-coder-30b)"
curl "$ANTHROPIC_BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" -H "content-type: application/json" \
  -d '{"model": "qwen3-coder", "messages": [{"role": "user", "content": "Hi"}]}'
```

## Cursor

Settings, Models: enable "Override OpenAI Base URL" with `<url>/v1`, paste the key as the OpenAI
API key, and add a custom model named `qwen3-coder`. Cursor sends requests from its own servers, so
the URL has to be public (Modal's is; a local Docker deployment isn't).

## Testing a deployment

```bash
make live NAME=qwen3-coder-30b
```

runs [`tests/test_live.py`](../tests/test_live.py) against it: model listing, auth, chat,
streaming, structured outputs, JSON mode and tool calls through both SDKs.
