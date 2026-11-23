"""Modal app for one deployment: a scale-to-zero SGLang server behind an always-on proxy server.

`modal deploy` runs a copy of this file outside the package, configured by BYOM_MODAL_SPEC. It may
only import the stdlib and modal, because Modal also imports it inside the SGLang image.
"""

import json
import os
import subprocess

import modal

SPEC_JSON = os.environ["BYOM_MODAL_SPEC"]
SPEC = json.loads(SPEC_JSON)
# Not /root/.cache/huggingface: the SGLang image already has files there and Modal only mounts
# volumes on empty paths.
CACHE = "/byom-cache"
CACHE_ENV = {
    "HF_HOME": CACHE,
    "HF_HUB_CACHE": f"{CACHE}/hub",
    # compiled kernels (DeepGEMM, Triton, FlashInfer); otherwise every cold start recompiles them
    "SGLANG_CACHE_DIR": f"{CACHE}/sglang",
}

app = modal.App(SPEC["app"])
secrets = [modal.Secret.from_name(SPEC["app"])]
cache = modal.Volume.from_name("byom-hf-cache", create_if_missing=True)

proxy_image = modal.Image.debian_slim(python_version="3.12")
if SPEC["local_root"]:
    proxy_image = proxy_image.add_local_dir(
        SPEC["local_root"], "/src", copy=True,
        ignore=[".git", ".venv", "**/__pycache__", "docs", "tests"],
    ).run_commands("pip install --no-cache-dir /src")
else:
    proxy_image = proxy_image.uv_pip_install(f"byom=={SPEC['version']}")
proxy_image = proxy_image.uv_pip_install(
    f"modal=={SPEC['modal_version']}", "huggingface_hub[hf_xet]"
).env({**CACHE_ENV, "BYOM_MODAL_SPEC": SPEC_JSON})

sglang_image = (
    modal.Image.from_registry(SPEC["image"])
    .entrypoint([])
    .env({**CACHE_ENV, "HF_XET_HIGH_PERFORMANCE": "1", "BYOM_MODAL_SPEC": SPEC_JSON})
)


@app.server(
    image=sglang_image,
    gpu=SPEC["gpu"],
    port=30000,
    secrets=secrets,
    volumes={CACHE: cache},
    startup_timeout=SPEC["startup_s"],
    scaledown_window=SPEC["idle_s"],
    max_containers=1,
    routing_region=SPEC["region"],
    unauthenticated=True,
)
class SGLang:
    @modal.enter()
    def start(self) -> None:
        self.proc = subprocess.Popen(
            [*SPEC["command"], "--api-key", os.environ["BYOM_API_KEY"]]
        )


@app.server(
    image=proxy_image,
    port=8080,
    min_containers=1,
    max_containers=1,
    startup_timeout=120,
    routing_region=SPEC["region"],
    unauthenticated=True,
)
class Proxy:
    @modal.enter()
    def start(self) -> None:
        upstream = modal.Server.from_name(SPEC["app"], "SGLang").get_url()
        if not upstream:
            raise RuntimeError("could not resolve the SGLang server URL")
        cmd = ["byom", "serve", "--upstream", upstream, "--host", "0.0.0.0",
               "--port", "8080", "--hold", str(SPEC["startup_s"])]
        if SPEC["drop_thinking"]:
            cmd.append("--drop-thinking")
        self.proc = subprocess.Popen(cmd)


@app.function(image=proxy_image, secrets=secrets, volumes={CACHE: cache}, timeout=3600)
def prefetch() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(SPEC["model"], revision=SPEC["revision"])
    cache.commit()
