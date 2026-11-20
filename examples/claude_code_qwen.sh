#!/usr/bin/env bash
# Deploy Qwen3-Coder-30B on a Modal H100 and have Claude Code build a small tested program with it.
#
#   pip install 'sgbridge[modal]' && modal setup
#   bash examples/claude_code_qwen.sh
#
# The GPU scales to zero after 20 idle minutes; `sgbridge down qwen3-coder-30b` removes everything.
set -euo pipefail

NAME=qwen3-coder-30b

if ! sgbridge env "$NAME" >/dev/null 2>&1; then
  sgbridge deploy modal --profile "$NAME"
fi
eval "$(sgbridge env "$NAME")"

work=$(mktemp -d)
cd "$work"
echo "working in $work (the first request may wait a few minutes for the GPU to boot)"

# Skipping permission prompts is fine in a throwaway directory; don't do this in a real repo.
claude -p --dangerously-skip-permissions "
Create wordfreq.py: a command line tool that prints the 10 most common words in a text file,
case-insensitive and ignoring punctuation, as 'count word' lines. Add test_wordfreq.py with unittest
tests, then run 'python3 -m unittest -v' and fix any failures until they pass. Finish with a one-line
summary."

echo
ls "$work"
