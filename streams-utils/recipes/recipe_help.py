"""CLI helper for recipe manifests.

This script parses a YAML manifest describing recipes and their
corresponding flag lists. The manifest is expected to have the
following structure::

    recipes:
      recipe-name:
        summary: Short summary of the recipe
        types:
          type-name:
            subtypes:
              subtype-name:
                flags:
                  - --flag

Usage:
The CLI provides two explicit subcommands under the ``just help`` entry
point and a positional helper:

``just help list``
    Show all recipes and their summaries.

``just help tree``
    Show recipes as a RECIPE | TYPE | SUBTYPE table.

``just help <recipe> <type> <subtype>``
    Emit the fully formatted flag list for the requested recipe.

"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import typer
import yaml

app = typer.Typer(help="Inspect recipe YAML manifests")

DEFAULT_MANIFEST = Path(__file__).with_name("recipes.yaml")
INDENT = "        "



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


def iter_recipe_types(info: Dict[str, dict]) -> Iterable[tuple[str, str]]:
    """Yield (type, subtype) tuples for a recipe."""
    types = info.get("types") or {}
    if types:
        for type_name, type_info in types.items():
            subtypes = type_info.get("subtypes") or {}
            if subtypes:
                for subtype_name in subtypes.keys():
                    yield type_name, subtype_name
            else:
                yield type_name, ""
        return

    tree_rows = info.get("tree_rows") or []
    if not tree_rows:
        yield "", ""
        return

    for row in tree_rows:
        yield str(row.get("type", "") or ""), str(row.get("subtype", "") or "")


def resolve_flags(
    manifest: Dict[str, dict],
    recipe: str,
    type_name: str,
    subtype_name: str,
) -> List[str]:
    """Return the list of flags for a recipe/type/subtype."""
    recipes = manifest.get("recipes", {})
    if recipe not in recipes:
        available = ", ".join(sorted(recipes.keys()))
        raise typer.BadParameter(
            f"Unknown recipe '{recipe}'. Available recipes: {available}"
        )

    info = recipes[recipe]
    types = info.get("types") or {}
    if not types:
        return []

    if type_name not in types:
        available = ", ".join(sorted(types.keys()))
        raise typer.BadParameter(
            f"Unknown type '{type_name}' for recipe '{recipe}'. "
            f"Available types: {available}"
        )

    type_info = types[type_name]
    subtypes = type_info.get("subtypes") or {}
    if not subtypes:
        if subtype_name:
            raise typer.BadParameter(
                f"Recipe '{recipe}' type '{type_name}' does not define subtypes."
            )
        return type_info.get("flags") or []

    if subtype_name not in subtypes:
        available = ", ".join(sorted(subtypes.keys()))
        raise typer.BadParameter(
            f"Unknown subtype '{subtype_name}' for recipe '{recipe}' type '{type_name}'. "
            f"Available subtypes: {available}"
        )

    subtype_info = subtypes[subtype_name]
    return subtype_info.get("flags") or []


def format_flags(flags: List[str]) -> str:
    """Format flags for inclusion in the justfile recipes."""
    if not flags:
        return f"{INDENT}# (no flags)"

    formatted: List[str] = []
    for flag in flags:
        stripped = flag.lstrip()
        if stripped.startswith("#"):
            formatted.append(f"{INDENT}{flag}")
        else:
            formatted.append(f"{INDENT}{flag} \\")
    return "\n".join(formatted)


def render_flags(
    manifest: Path,
    recipe: str,
    type_name: str,
    subtype_name: str,
) -> None:
    """Load data and emit formatted flags for the requested recipe."""
    data = load_manifest(manifest)
    flags = resolve_flags(data, recipe, type_name, subtype_name)
    typer.echo(format_flags(flags))


@app.callback(
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def handle_cli(ctx: typer.Context, manifest: Path = DEFAULT_MANIFEST) -> None:
    """Handle positional recipe queries."""
    if ctx.invoked_subcommand:
        return

    if not ctx.args:
        typer.echo(ctx.get_help())
        return

    if len(ctx.args) != 3:
        raise typer.BadParameter(
            "Expected '<recipe> <type> <subtype>' or an explicit subcommand."
        )

    recipe, type_name, subtype_name = ctx.args
    render_flags(manifest, recipe, type_name, subtype_name)



@app.command("list")
def list_recipes(manifest: Path = DEFAULT_MANIFEST) -> None:
    """List recipes and their summaries."""
    data = load_manifest(manifest)
    for name, info in iter_flag_groups(data):
        summary = info.get("summary", "")
        typer.echo(f"{name}: {summary}")



@app.command("tree")
def tree(manifest: Path = DEFAULT_MANIFEST) -> None:
    """Show recipes as a 3-column table: RECIPE | TYPE | SUBTYPE."""
    data = load_manifest(manifest)

    # Build rows: (recipe, type, subtype)
    rows: List[tuple[str, str, str]] = []
    for recipe, info in iter_flag_groups(data):
        for type_name, subtype_name in iter_recipe_types(info):
            rows.append((recipe, type_name, subtype_name))

    # Header
    header = ("RECIPE", "TYPE", "SUBTYPE")
    all_rows = [header, *rows]

    # Compute column widths
    w0 = max(len(r[0]) for r in all_rows)
    w1 = max(len(r[1]) for r in all_rows)
    w2 = max(len(r[2]) for r in all_rows)

    def fmt(r: tuple[str, str, str]) -> str:
        return f"{r[0]:<{w0}}  {r[1]:<{w1}}  {r[2]:<{w2}}"

    typer.echo(fmt(header))
    typer.echo(f"{'-'*w0}  {'-'*w1}  {'-'*w2}")

    for r in rows:
        typer.echo(fmt(r))
    print("""
    For a comprehensive set of flags: just help <RECIPE> <TYPE> <SUBTYPE>
    For detailed info on a recipe: just help <RECIPE>
    For detailed info on a flag: just help <FLAG>
    """)

def main() -> None:
    """Fallback entrypoint for raw argv parsing."""
    args = sys.argv[1:]
    if not args or args[0] in {"list", "tree"}:
        app()
        return

    recipe = args[0]
    type_name = args[1] if len(args) > 1 else ""
    subtype_name = args[2] if len(args) > 2 else ""
    render_flags(DEFAULT_MANIFEST, recipe, type_name, subtype_name)

if __name__ == "__main__":
    main()
