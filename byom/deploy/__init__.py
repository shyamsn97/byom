"""Deployments (SGLang + proxy on Docker or Modal), each saved as a small JSON record."""

from __future__ import annotations

import json
import os
import secrets
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", "~/.local/state")).expanduser() / "byom"
REPO_ROOT = Path(__file__).resolve().parents[2]


class DeployError(Exception):
    pass


@dataclass
class Deployment:
    name: str
    target: str
    url: str
    api_key: str = field(repr=False)
    model: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return STATE_DIR / self.name

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "deployment.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, name: str) -> Deployment:
        path = STATE_DIR / name / "deployment.json"
        if not path.is_file():
            have = ", ".join(names()) or "none"
            raise DeployError(f"no deployment named {name!r} (have: {have})")
        return cls(**json.loads(path.read_text()))

    def remove(self) -> None:
        (self.dir / "deployment.json").unlink(missing_ok=True)

    def claude_code_env(self) -> str:
        env = {
            "ANTHROPIC_BASE_URL": self.url,
            "ANTHROPIC_AUTH_TOKEN": self.api_key,
            "ANTHROPIC_MODEL": self.model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": self.model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": self.model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": self.model,
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "API_TIMEOUT_MS": "3000000",
        }
        return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items())


def names() -> list[str]:
    if not STATE_DIR.is_dir():
        return []
    return sorted(p.parent.name for p in STATE_DIR.glob("*/deployment.json"))


def api_key_for(name: str) -> str:
    """Reuse the key on redeploy so clients keep working."""
    try:
        return Deployment.load(name).api_key
    except DeployError:
        return "sk-byom-" + secrets.token_urlsafe(24)


def local_checkout() -> Path | None:
    """The repo root when running from a git checkout (the proxy is then built from source)."""
    return REPO_ROOT if (REPO_ROOT / "docker" / "proxy.Dockerfile").is_file() else None
