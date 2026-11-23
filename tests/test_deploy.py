from __future__ import annotations

import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from byom.deploy import Deployment, api_key_for
from byom.deploy.docker import render_compose
from byom.profiles import list_profiles, load_profile


def test_bundled_profiles_load() -> None:
    assert "qwen3-coder-30b" in list_profiles()
    p = load_profile("qwen3-coder-30b")
    cmd = p.sglang_command()
    assert cmd[cmd.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert cmd[cmd.index("--served-model-name") + 1] == "qwen3-coder"
    assert "--tp" not in cmd
    with pytest.raises(ValueError, match="unknown profile"):
        load_profile("nope")


def test_multi_gpu_adds_tp(tmp_path: Path) -> None:
    path = tmp_path / "big.yaml"
    path.write_text(yaml.safe_dump({"name": "big", "model": "m", "revision": "r", "image": "i",
                                    "served_name": "big", "gpu": "H100:4"}))
    cmd = load_profile(str(path)).sglang_command()
    assert cmd[cmd.index("--tp") + 1] == "4"


def test_compose(tmp_path: Path) -> None:
    compose = render_compose(load_profile("qwen3-coder-30b"), port=9000)
    sglang, proxy = compose["services"]["sglang"], compose["services"]["proxy"]
    assert sglang["command"][-2:] == ["--api-key", "${BYOM_API_KEY}"]
    assert proxy["ports"] == ["127.0.0.1:9000:8080"]
    assert "--drop-thinking" in proxy["command"]
    assert proxy["command"][proxy["command"].index("--upstream") + 1] == "http://sglang:30000"
    if shutil.which("docker"):
        (tmp_path / "compose.yaml").write_text(yaml.safe_dump(compose))
        (tmp_path / ".env").write_text("BYOM_API_KEY=x\n")
        result = subprocess.run(["docker", "compose", "--project-directory", str(tmp_path),
                                 "config", "--quiet"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_deployment_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("byom.deploy.STATE_DIR", tmp_path)
    key = api_key_for("box")
    assert key.startswith("sk-byom-")
    Deployment("box", "docker", "http://127.0.0.1:8080", key, "qwen3-coder").save()
    assert (tmp_path / "box" / "deployment.json").stat().st_mode & 0o777 == 0o600
    assert api_key_for("box") == key
    env = Deployment.load("box").claude_code_env()
    assert f"ANTHROPIC_AUTH_TOKEN={key}" in env and "ANTHROPIC_MODEL=qwen3-coder" in env


def test_modal_app_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("modal")
    p = load_profile("qwen3-coder-30b")
    monkeypatch.setenv("BYOM_MODAL_SPEC", json.dumps({
        "app": "byom-test", "image": p.image, "command": p.sglang_command(), "gpu": p.gpu,
        "model": p.model, "revision": p.revision, "startup_s": 1200, "idle_s": 1200,
        "drop_thinking": True, "region": "us-east", "local_root": None, "version": "0.1.0",
        "modal_version": "1.6.4",
    }))
    app_file = Path(__file__).parents[1] / "byom" / "deploy" / "modal" / "app.py"
    imports = {line.split()[1].split(".")[0] for line in app_file.read_text().splitlines()
               if line.startswith(("import ", "from "))}
    assert imports <= {"json", "os", "subprocess", "modal"}
    assert runpy.run_path(str(app_file))["app"].name == "byom-test"
