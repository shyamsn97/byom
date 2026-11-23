"""Deploy to Modal: create the secret, `modal deploy` app.py, download the weights."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

from byom import __version__
from byom.deploy import DeployError, Deployment, api_key_for, local_checkout
from byom.profiles import Profile

# Older Modal image builders install modal's own deps (pydantic 1.x) into registry images,
# which breaks SGLang.
IMAGE_BUILDER_VERSION = "2026.10"


def up(profile: Profile, name: str, *, region: str = "us-east") -> Deployment:
    try:
        import modal
    except ImportError:
        raise DeployError("the modal package is not installed: pip install 'byom[modal]'") \
            from None
    app_name = f"byom-{name}"
    api_key = api_key_for(name)
    secret = {"BYOM_API_KEY": api_key}
    if token := os.environ.get("HF_TOKEN"):
        secret["HF_TOKEN"] = token
    modal.Secret.objects.delete(app_name, allow_missing=True)
    modal.Secret.objects.create(app_name, secret)

    checkout = local_checkout()
    spec = {
        "app": app_name,
        "image": profile.image,
        "command": profile.sglang_command(),
        "gpu": profile.gpu,
        "model": profile.model,
        "revision": profile.revision,
        "startup_s": profile.startup_timeout_min * 60,
        "idle_s": profile.idle_timeout_min * 60,
        "drop_thinking": profile.drop_thinking,
        "region": region,
        "local_root": str(checkout) if checkout else None,
        "version": __version__,
        "modal_version": modal.__version__,
    }
    with tempfile.TemporaryDirectory() as tmp:
        app_file = Path(tmp) / f"byom_{name.replace('-', '_')}.py"
        shutil.copy(Path(__file__).with_name("app.py"), app_file)
        env = {**os.environ, "BYOM_MODAL_SPEC": json.dumps(spec),
               "MODAL_IMAGE_BUILDER_VERSION": IMAGE_BUILDER_VERSION}
        if subprocess.run([sys.executable, "-m", "modal", "deploy", str(app_file)],
                          env=env).returncode != 0:
            raise DeployError("modal deploy failed")
    print("downloading weights to the byom-hf-cache volume...", flush=True)
    modal.Function.from_name(app_name, "prefetch").remote()
    url = modal.Server.from_name(app_name, "Proxy").get_url()
    if not url:
        raise DeployError(f"could not find the URL of {app_name}/Proxy")
    dep = Deployment(name, "modal", url.rstrip("/"), api_key, profile.served_name,
                     {"app": app_name})
    dep.save()
    return dep


def down(dep: Deployment) -> None:
    app = dep.extra["app"]
    subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes", app], check=False)
    import modal

    modal.Secret.objects.delete(app, allow_missing=True)


def logs(dep: Deployment, follow: bool = False) -> Iterator[str]:
    cmd = [sys.executable, "-m", "modal", "app", "logs", dep.extra["app"],
           *(["-f"] if follow else [])]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True) as proc:
        assert proc.stdout is not None
        yield from proc.stdout
