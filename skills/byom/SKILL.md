---
name: byom
description: Deploy an open model with SGLang on Modal or Docker using the byom CLI, and connect Claude Code, the OpenAI or Anthropic SDKs, Cursor or other agents to it. Use when the user wants to run Claude Code or a coding agent on an open or self-hosted model, deploy SGLang, add a byom model profile, or debug a byom deployment.
---

# byom

byom deploys SGLang plus a small proxy and gives back one URL that serves both the Anthropic
Messages API (at `<url>`) and the OpenAI API (at `<url>/v1`), with an `sk-byom-...` key.

## Deploy

```bash
pip install 'byom[modal]' && modal setup          # Modal: scale-to-zero GPU, public URL
byom deploy modal --profile qwen3-coder-30b

byom deploy docker --profile qwen3-coder-30b      # Docker: needs an NVIDIA host
byom deploy docker --profile qwen3-coder-30b --context my-gpu-box   # remote Docker context
```

Other commands: `byom list`, `byom env NAME`, `byom logs NAME -f`, `byom down NAME`.
Set `HF_TOKEN` before deploying gated models. Docker can't run on macOS without `--context`.

The deployment name defaults to the profile name. `deploy` prints the Claude Code settings.

## Connect

Claude Code:

```bash
eval "$(byom env NAME)" && claude
```

Python (the saved deployment has `.url`, `.api_key`, `.model`):

```python
from byom.deploy import Deployment
from openai import OpenAI

dep = Deployment.load("NAME")
client = OpenAI(base_url=f"{dep.url}/v1", api_key=dep.api_key, timeout=900)
```

Anthropic SDK: `Anthropic(base_url=dep.url, api_key=dep.api_key, timeout=900)`.
Streaming, structured outputs (`client.chat.completions.parse(response_format=Model)`), JSON mode
and tool calls all work. The model name is the profile's `served_name` (e.g. `qwen3-coder`), never
a `claude-*` name.

Cursor and other OpenAI clients: base URL `<url>/v1`, the `sk-byom-` key, model `served_name`.
Cursor calls from its own servers, so only a public URL works (Modal's is; local Docker isn't).

## Profiles

A profile is YAML; pass a path with `--profile path/to/model.yaml`:

```yaml
name: qwen3-coder-30b
model: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
revision: dcaee4d4dfc5ee71ad501f01f530e5652438fde0   # pin a commit
image: lmsysorg/sglang:v0.5.23-runtime
served_name: qwen3-coder
gpu: H100:1                 # H100:2 etc. adds --tp automatically
args:                       # any sglang.launch_server flag, underscores for dashes
  tool_call_parser: qwen3_coder
  context_length: 131072
drop_thinking: true
```

- Coding agents need `tool_call_parser` set to the parser SGLang has for that model family.
- Set `drop_thinking: true` unless `args` sets a `reasoning_parser`; Claude Code always sends
  `thinking`, and SGLang rejects it otherwise.
- The model must be on the Hugging Face Hub; local checkpoints aren't supported.
- Only text-generating models SGLang supports can be served. Models that need their own
  Transformers code (for example LiquidAI's d1 decision models) can't; see
  `examples/d1_snake.py` for running one directly on Modal.

## Behavior to expect

- On Modal the GPU scales to zero after 20 idle minutes. The next request waits 2-4 minutes for
  it to boot instead of failing, so use long client timeouts. A fresh deploy also cold-starts.
- The always-on CPU proxy costs a little; `byom down NAME` removes everything.
- Claude Code prints "doesn't recognize the model ... assumes a context window of 200k"; that's
  harmless.

## Troubleshooting

- 401: wrong key; `byom env NAME` shows the current one.
- 502 "upstream is not reachable": SGLang didn't come up; check `byom logs NAME`.
- 400 mentioning `thinking`: the profile needs `drop_thinking: true`.
- Tool calls come back as plain text: wrong or missing `tool_call_parser`.

To verify a deployment end to end from a byom checkout: `make live NAME=<deployment>`.
