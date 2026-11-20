from __future__ import annotations

from types import ModuleType
from typing import Annotated

import typer

from sgbridge import proxy
from sgbridge.deploy import DeployError, Deployment, names
from sgbridge.profiles import list_profiles, load_profile

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False,
                  help="Deploy SGLang and put Claude Code (and friends) in front of it.")


@app.command()
def serve(
    upstream: Annotated[str, typer.Option(help="SGLang URL, e.g. http://localhost:30000")],
    host: str = "127.0.0.1",
    port: int = 8080,
    drop_thinking: Annotated[bool, typer.Option(
        help="Drop `thinking` (for models served without a reasoning parser).")] = False,
    hold: Annotated[float, typer.Option(
        help="Seconds to retry while the upstream is starting (503 / refused).")] = 0,
) -> None:
    """Run the proxy in front of an SGLang server."""
    proxy.serve(upstream, host=host, port=port, drop_thinking=drop_thinking, hold_s=hold)


@app.command()
def deploy(
    target: Annotated[str, typer.Argument(help="docker or modal")],
    profile: Annotated[str, typer.Option(help="Profile name or path to a .yaml")],
    name: Annotated[str | None, typer.Option(help="Defaults to the profile name.")] = None,
    context: Annotated[str | None, typer.Option(help="docker: remote Docker context.")] = None,
    port: Annotated[int, typer.Option(help="docker: host port for the proxy.")] = 8080,
    public: Annotated[bool, typer.Option(help="docker: listen on 0.0.0.0.")] = False,
    region: Annotated[str, typer.Option(help="modal: routing region.")] = "us-east",
) -> None:
    """Deploy SGLang + the proxy, then print the Claude Code settings."""
    prof = load_profile(profile)
    name = name or prof.name
    if target == "docker":
        from sgbridge.deploy import docker

        dep = docker.up(prof, name, context=context, port=port, public=public)
    elif target == "modal":
        from sgbridge.deploy import modal

        dep = modal.up(prof, name, region=region)
    else:
        raise DeployError(f"unknown target {target!r}; use docker or modal")
    typer.secho(f"\n{dep.name} is up at {dep.url}", fg="green")
    typer.echo("The first request after idle waits for the GPU to start (a few minutes).")
    typer.echo("Claude Code settings (also: `sgbridge env " + dep.name + "`):\n")
    typer.echo(dep.claude_code_env())


@app.command()
def env(name: str) -> None:
    """Print the Claude Code environment for a deployment."""
    typer.echo(Deployment.load(name).claude_code_env())


@app.command()
def down(name: str) -> None:
    """Stop a deployment and forget it."""
    dep = Deployment.load(name)
    _target(dep).down(dep)
    dep.remove()
    typer.echo(f"stopped {name}")


@app.command()
def logs(name: str, follow: Annotated[bool, typer.Option("--follow", "-f")] = False) -> None:
    """Show deployment logs."""
    dep = Deployment.load(name)
    for line in _target(dep).logs(dep, follow):
        typer.echo(line, nl=False)


@app.command("list")
def list_() -> None:
    """Saved deployments and bundled profiles."""
    for n in names():
        dep = Deployment.load(n)
        typer.echo(f"{n:24} {dep.target:7} {dep.url}")
    typer.echo(f"profiles: {', '.join(list_profiles())}")


def _target(dep: Deployment) -> ModuleType:
    if dep.target == "docker":
        from sgbridge.deploy import docker

        return docker
    from sgbridge.deploy import modal

    return modal


def main() -> None:
    try:
        app()
    except (DeployError, ValueError) as e:
        typer.secho(f"error: {e}", fg="red", err=True)
        raise SystemExit(1) from None
