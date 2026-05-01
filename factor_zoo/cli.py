"""Click CLI entry points for Factor Zoo."""
from __future__ import annotations

import json
import sys

import click


@click.group()
def app() -> None:
    """Factor Zoo — browse, screen, and analyze 200+ equity factors."""


@app.command("list")
@click.option("--category", default=None, help="Filter by category (e.g. momentum).")
@click.option("--min-sharpe", type=float, default=None, help="Minimum annualized Sharpe.")
@click.option("--min-t-stat", type=float, default=None, help="Minimum t-statistic.")
@click.option("--limit", type=int, default=20, show_default=True, help="Max rows to show.")
def list_factors(category, min_sharpe, min_t_stat, limit):
    """List factors matching the given filters."""
    from factor_zoo import FactorZoo

    try:
        fz = FactorZoo()
    except Exception as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    df = fz.list_factors(category=category, min_sharpe=min_sharpe, min_t_stat=min_t_stat)
    cols = ["id", "name", "category", "sharpe", "t_stat", "ann_return", "ann_vol"]
    cols = [c for c in cols if c in df.columns]
    click.echo(df[cols].head(limit).to_string(index=False))
    fz.close()


@app.command("detail")
@click.argument("factor_id")
def detail(factor_id):
    """Show stats and replication score for FACTOR_ID."""
    from factor_zoo import FactorZoo

    try:
        fz = FactorZoo()
    except Exception as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    try:
        stats = fz.get_stats(factor_id)
    except KeyError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    click.echo(f"\n{factor_id}\n" + "─" * len(factor_id))
    for k, v in stats.items():
        if isinstance(v, float):
            click.echo(f"  {k:<25} {v:.4f}")
        else:
            click.echo(f"  {k:<25} {v}")

    rep = fz.replication_score(factor_id)
    click.echo("\nReplication:")
    for k, v in rep.items():
        if isinstance(v, float):
            click.echo(f"  {k:<25} {v:.4f}")
        else:
            click.echo(f"  {k:<25} {v}")

    fz.close()


@app.command("update")
@click.option("--check-only", is_flag=True, default=False, help="Check without rebuilding.")
def update(check_only):
    """Check OSAP for new factors and optionally rebuild the database."""
    from factor_zoo import FactorZoo

    try:
        fz = FactorZoo()
    except Exception as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    result = fz.update(check_only=check_only)

    if result.get("error"):
        click.echo(f"Warning: {result['error']}", err=True)

    new = result.get("new_signals", [])
    if not result.get("update_available"):
        click.echo("Already up to date.")
    elif check_only:
        click.echo(f"{len(new)} new signal(s) available: {', '.join(new[:10])}")
        if len(new) > 10:
            click.echo(f"  … and {len(new) - 10} more")
        click.echo("Run `factorzoo update` (without --check-only) to rebuild.")
    else:
        click.echo(f"Rebuilt database with {len(new)} new signal(s).")

    fz.close()


@app.command("build")
def build():
    """Force a full local rebuild from OSAP + Ken French Data Library."""
    import os
    import sys as _sys

    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    _sys.path.insert(0, os.path.abspath(scripts_dir))

    try:
        import build_db
    except ImportError:
        click.echo("Could not import scripts/build_db.py.", err=True)
        sys.exit(1)

    try:
        build_db.main()
        click.echo("Database rebuilt successfully.")
    except Exception as exc:
        click.echo(f"Build failed: {exc}", err=True)
        sys.exit(1)


@app.command("zoo-summary")
def zoo_summary():
    """Print aggregate statistics about the factor zoo."""
    from factor_zoo import FactorZoo

    try:
        fz = FactorZoo()
    except Exception as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    summary = fz.zoo_summary()
    click.echo(json.dumps(summary, indent=2))
    fz.close()
