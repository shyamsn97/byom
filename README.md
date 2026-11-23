# byom

Bring your own model to Claude Code and other coding agents. byom deploys
[SGLang](https://github.com/sgl-project/sglang) on Modal or Docker and puts a small proxy in front
of it that smooths over the few places where Claude Code and SGLang disagree.

```bash
pip install 'byom[modal]'
byom deploy modal --profile qwen3-coder-30b
```

`deploy` prints the environment variables for Claude Code:

```bash
export ANTHROPIC_BASE_URL=https://<you>--byom-qwen3-coder-30b-proxy.us-east.modal.direct
export ANTHROPIC_AUTH_TOKEN=sk-byom-...
export ANTHROPIC_MODEL=qwen3-coder
...
claude
```

Requests go straight from Claude Code to your deployment, so nothing goes to Anthropic. You pay
only for the GPU.

## Targets

**Modal** (`byom deploy modal`): the GPU server scales to zero after 20 idle minutes, and a
small always-on CPU proxy keeps the URL stable. The first request after idle waits for the GPU to
boot (about 4 minutes with cached weights) instead of failing. Needs `modal setup` first.

**Docker** (`byom deploy docker`): SGLang and the proxy as a Compose project on an NVIDIA
machine, reachable at `http://127.0.0.1:8080`. Use `--context <docker context>` to deploy to a
remote GPU box from your laptop.

**Your own SGLang server**: `byom serve --upstream http://localhost:30000 --drop-thinking`.

```bash
byom list          # deployments and profiles
byom env NAME      # print the Claude Code settings again
byom logs NAME -f
byom down NAME
```

Gated models need `HF_TOKEN` set when you deploy.

## What the proxy changes

Everything is forwarded to SGLang byte for byte, except:

- `thinking` is dropped for models served without a reasoning parser (Claude Code always sends
  it; SGLang rejects it).
- Claude Code's per-request attribution line is removed from the system prompt so SGLang's prefix
  cache works.
- `x-api-key` is also sent as `Authorization: Bearer`, which is what SGLang checks.
- While the backend is starting, 503s and refused connections are retried instead of returned.

## Profiles

A profile is a YAML file naming the model, the SGLang image and flags, and the GPU:

```yaml
name: qwen3-coder-30b
model: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
revision: dcaee4d4dfc5ee71ad501f01f530e5652438fde0
image: lmsysorg/sglang:v0.5.23-runtime
served_name: qwen3-coder
gpu: H100:1
args:
  tool_call_parser: qwen3_coder
  context_length: 131072
drop_thinking: true
```

Pass your own with `--profile path/to/model.yaml`. The model needs an SGLang
`--tool-call-parser`, since Claude Code does everything through tool calls.

## Using it from code

The same URL serves the OpenAI API under `/v1`, so the OpenAI SDK, Cursor and most other tools work
too, including streaming, structured outputs and tool calls:

```python
from openai import OpenAI
from byom.deploy import Deployment

dep = Deployment.load("qwen3-coder-30b")
client = OpenAI(base_url=f"{dep.url}/v1", api_key=dep.api_key, timeout=900)

for chunk in client.chat.completions.create(
    model=dep.model, stream=True,
    messages=[{"role": "user", "content": "Explain Python's GIL in two sentences."}],
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

[docs/usage.md](https://github.com/shyamsn97/byom/blob/main/docs/usage.md) covers the OpenAI and Anthropic SDKs, structured outputs, tool
calls, curl and Cursor. Runnable versions are in [examples/](https://github.com/shyamsn97/byom/tree/main/examples):

- [`claude_code_qwen.sh`](https://github.com/shyamsn97/byom/blob/main/examples/claude_code_qwen.sh): deploy, then have Claude Code write and
  test a small program.
- [`openai_sdk.py`](https://github.com/shyamsn97/byom/blob/main/examples/openai_sdk.py): chat, streaming, structured outputs, JSON mode, tool
  calls.
- [`anthropic_sdk.py`](https://github.com/shyamsn97/byom/blob/main/examples/anthropic_sdk.py): messages, streaming, tool use.
- [`d1_snake.py`](https://github.com/shyamsn97/byom/blob/main/examples/d1_snake.py): LiquidAI's
  d1-2B decision model playing snake from rendered frames on a Modal GPU
  (`modal run examples/d1_snake.py`). d1 doesn't generate text, so it runs in Transformers
  rather than through byom.

`make live NAME=<deployment>` runs all of these checks against a deployment as tests.

## Agent skill

[`skills/byom/SKILL.md`](https://github.com/shyamsn97/byom/blob/main/skills/byom/SKILL.md) teaches
coding agents to deploy and connect to byom. Copy the `skills/byom` folder into
`~/.claude/skills/` for Claude Code or `~/.cursor/skills/` for Cursor.

## Development

```bash
make install   # uv sync
make check     # ruff, mypy, pytest
make live NAME=qwen3-coder-30b   # end-to-end tests against a real deployment
make bump PART=minor             # bump the version in pyproject.toml (patch, minor or major)
```

To release, commit the bump and publish a GitHub release tagged `v<version>`; the publish
workflow checks the tag matches and uploads to PyPI.
