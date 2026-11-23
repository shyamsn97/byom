"""SGLang + proxy as a Docker Compose project, on this machine or a remote Docker context."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from byom import __version__
from byom.deploy import (
    DeployError,
    Deployment,
    api_key_for,
    local_checkout,
)
from byom.profiles import Profile

HF_CACHE_VOLUME = "byom-hf-cache"


def render_compose(profile: Profile, *, port: int = 8080, public: bool = False,
                   hf_cache: str = HF_CACHE_VOLUME,
                   build_root: Path | None = None) -> dict[str, Any]:
    proxy_cmd = ["byom", "serve", "--upstream", "http://sglang:30000", "--host", "0.0.0.0",
                 "--port", "8080", "--hold", str(profile.startup_timeout_min * 60)]
    if profile.drop_thinking:
        proxy_cmd.append("--drop-thinking")
    proxy: dict[str, Any] = {
        "command": proxy_cmd,
        "ports": [f"{'0.0.0.0' if public else '127.0.0.1'}:{port}:8080"],
        "restart": "unless-stopped",
    }
    if build_root:
        proxy["build"] = {"context": str(build_root), "dockerfile": "docker/proxy.Dockerfile"}
    else:
        proxy["build"] = {"context": ".", "dockerfile_inline": (
            f"FROM python:3.12-slim\nRUN pip install --no-cache-dir byom=={__version__}\n")}
    compose: dict[str, Any] = {
        "services": {
            "sglang": {
                "image": profile.image,
                "command": [*profile.sglang_command(), "--api-key", "${BYOM_API_KEY}"],
                "env_file": [".env"],
                "volumes": [f"{hf_cache}:/root/.cache/huggingface"],
                "ipc": "host",
                "shm_size": "32g",
                "deploy": {"resources": {"reservations": {"devices": [
                    {"driver": "nvidia", "count": profile.gpu_count, "capabilities": ["gpu"]},
                ]}}},
                "restart": "unless-stopped",
            },
            "proxy": proxy,
        },
    }
    if hf_cache == HF_CACHE_VOLUME:
        compose["volumes"] = {HF_CACHE_VOLUME: {"name": HF_CACHE_VOLUME}}
    return compose


def up(profile: Profile, name: str, *, context: str | None = None, port: int = 8080,
       public: bool = False) -> Deployment:
    if context is None and sys.platform == "darwin":
        raise DeployError("Docker on macOS has no NVIDIA GPU; use --context <remote GPU host> "
                          "or deploy to modal")
    hf_cache = (str(Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser())
                if context is None else HF_CACHE_VOLUME)
    dep = Deployment(name, "docker", f"http://127.0.0.1:{port}", api_key_for(name),
                     profile.served_name, {"context": context})
    dep.save()
    compose = render_compose(profile, port=port, public=public, hf_cache=hf_cache,
                             build_root=local_checkout())
    (dep.dir / "compose.yaml").write_text(yaml.safe_dump(compose, sort_keys=False, width=1000))
    env_lines = [f"BYOM_API_KEY={dep.api_key}"]
    if token := os.environ.get("HF_TOKEN"):
        env_lines.append(f"HF_TOKEN={token}")
    fd = os.open(dep.dir / ".env", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(env_lines) + "\n")
    _compose(dep, "up", "-d", "--build", "--remove-orphans")
    return dep


def down(dep: Deployment) -> None:
    _compose(dep, "down", "--remove-orphans")


def logs(dep: Deployment, follow: bool = False) -> Iterator[str]:
    with subprocess.Popen([*_base(dep), "logs", "--no-color", *(["-f"] if follow else [])],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        assert proc.stdout is not None
        yield from proc.stdout


def _compose(dep: Deployment, *args: str) -> None:
    if subprocess.run([*_base(dep), *args]).returncode != 0:
        raise DeployError(f"docker compose {args[0]} failed")


def _base(dep: Deployment) -> list[str]:
    context = dep.extra.get("context")
    return ["docker", *(["--context", context] if context else []), "compose",
            "-p", f"byom-{dep.name}", "--project-directory", str(dep.dir)]
