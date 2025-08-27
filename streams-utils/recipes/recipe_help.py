"""CLI helper for recipe manifests.

This script parses a YAML manifest describing recipes and their
corresponding flag groups.  The manifest is expected to have the
following structure::

    recipes:
      recipe-name:
        summary: Short summary of the recipe
        flag_groups:
          - name: Group name
            description: Optional description
            flags:
              - name: --flag
                description: Flag description

The CLI provides three commands:

``recipe-help list``
    Show all recipes and their summaries.

``recipe-help show <recipe>``
    Display the flag groups and flags for ``<recipe>``.

``recipe-help find <flag>``
    List recipes where ``<flag>`` appears and the groups containing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import typer
import yaml

app = typer.Typer(help="Inspect recipe YAML manifests")

DEFAULT_MANIFEST = Path(__file__).with_name("recipes.yaml")


def load_manifest(path: Path) -> Dict[str, dict]:
    """Load YAML manifest and return the parsed data."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"Manifest file not found: {path}") from exc


def iter_flag_groups(manifest: Dict[str, dict]) -> Iterable[tuple[str, dict]]:
    """Yield recipe name and its info."""
    return manifest.get("recipes", {}).items()


@app.command("list")
def list_recipes(manifest: Path = DEFAULT_MANIFEST) -> None:
    """List recipes and their summaries."""
    data = load_manifest(manifest)
    for name, info in iter_flag_groups(data):
        summary = info.get("summary", "")
        typer.echo(f"{name}: {summary}")


@app.command()
def show(recipe: str, manifest: Path = DEFAULT_MANIFEST) -> None:
    """Show flag groups for a recipe."""
    data = load_manifest(manifest)
    recipe_info = data.get("recipes", {}).get(recipe)
    if recipe_info is None:
        raise typer.BadParameter(f"Unknown recipe: {recipe}")

    groups = recipe_info.get("flag_groups", [])
    for group in groups:
        group_name = group.get("name", "(no name)")
        desc = group.get("description", "")
        if desc:
            typer.echo(f"{group_name} - {desc}")
        else:
            typer.echo(group_name)
        for flag in group.get("flags", []):
            flag_name = flag.get("name")
            flag_desc = flag.get("description", "")
            if flag_name:
                typer.echo(f"  {flag_name}: {flag_desc}")


@app.command()
def find(flag: str, manifest: Path = DEFAULT_MANIFEST) -> None:
    """Find recipes using a flag."""
    data = load_manifest(manifest)
    target = flag if flag.startswith("--") else f"--{flag}"
    found = False
    for recipe_name, info in iter_flag_groups(data):
        for group in info.get("flag_groups", []):
            for f in group.get("flags", []):
                name = f.get("name")
                if name in (flag, target):
                    if not found:
                        found = True
                    group_name = group.get("name", "(no name)")
                    typer.echo(f"{recipe_name}: {group_name}")
    if not found:
        typer.echo(f"No recipes include flag {flag}")


if __name__ == "__main__":
    app()
