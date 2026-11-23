"""Model profiles: which model, which SGLang image and flags, what GPU."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Profile:
    name: str
    model: str
    revision: str
    image: str
    served_name: str
    gpu: str = "H100:1"
    args: dict[str, Any] = field(default_factory=dict)
    drop_thinking: bool = False
    startup_timeout_min: int = 20
    idle_timeout_min: int = 20

    @property
    def gpu_count(self) -> int:
        return int(self.gpu.partition(":")[2] or 1)

    def sglang_command(self, port: int = 30000) -> list[str]:
        cmd = ["python3", "-m", "sglang.launch_server", "--model-path", self.model,
               "--revision", self.revision, "--served-model-name", self.served_name,
               "--host", "0.0.0.0", "--port", str(port)]
        args = dict(self.args)
        if self.gpu_count > 1:
            args.setdefault("tp", self.gpu_count)
        for key, value in args.items():
            flag = "--" + key.replace("_", "-")
            if value is True:
                cmd.append(flag)
            elif value not in (False, None):
                cmd += [flag, str(value)]
        return cmd


def load_profile(name_or_path: str) -> Profile:
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml") and path.is_file():
        text = path.read_text()
    else:
        bundled = files(__package__).joinpath(f"{name_or_path}.yaml")
        if not bundled.is_file():
            raise ValueError(f"unknown profile {name_or_path!r} (available: "
                             f"{', '.join(list_profiles())}, or a path to a .yaml file)")
        text = bundled.read_text()
    return Profile(**yaml.safe_load(text))


def list_profiles() -> list[str]:
    return sorted(p.name.removesuffix(".yaml") for p in files(__package__).iterdir()
                  if p.name.endswith(".yaml"))
