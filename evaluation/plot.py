"""Plot the real and generated trajectories of each model: their paths, and where the points fall on the grid.

Run from the URBAN-SYN folder (needs matplotlib, which the dev container has):
    python -m evaluation.plot --model diffrntraj
    python -m evaluation.plot --model tstrajgen mobilitygpt
    python -m evaluation.plot --model all

Each run writes evaluation/results/<date>_<time>_plots_<models>/ with, for each model:
  <model>_trajectories.png  the trajectories (all, or a random sample): the real test set, then each generated group
  <model>_density.png       the share of points in each grid cell (the cells the global geographic JSD compares)
  overview_trajectories.png and overview_density.png 
  
The settings are in the plots: section of eval_config.yaml.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import inputs
from evaluation.convert import chosen_models, convert_model, model_network
from evaluation.converters import CONVERTERS
from evaluation.metrics.grid import OUTSIDE, Grid
from evaluation.metrics.trajset import TrajSet, sample_index

DEFAULT_OUT = Path(__file__).with_name("results")
DEFAULTS = {"sample": None, "dpi": 200, "columns": 4, "overview_real": None}
NAMES = {"difftraj": "DiffTraj", "diffrntraj": "Diff-RNTraj", "tstrajgen": "TS-TrajGen", "mobilitygpt": "MobilityGPT",
         "lstm_trajgan": "LSTM-TrajGAN", "urban_syn": "URBAN-SYN"}
REAL_COLOR, GENERATED_COLOR = "#3a3a3a", "#1f5fa8"
PANEL_WIDTH_IN = 3.6


@dataclass
class Panel:
    """One map panel: a set of trajectories and how to label it."""

    title: str
    trajectories: TrajSet
    unit: str
    real: bool


def parse_args(argv=None) -> argparse.Namespace:
    """Read the command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot real and generated trajectories.")
    parser.add_argument("--model", nargs="+", required=True,
                        help=f"model name(s) to plot, or 'all': {', '.join(CONVERTERS)}")
    parser.add_argument("--config", default=inputs.DEFAULT_CONFIG, help="path to eval_config.yaml")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="folder for the plots (default: evaluation/results); each run gets a subfolder")
    return parser.parse_args(argv)


def plot_settings(config: dict) -> dict:
    """The plots: section of the config, with defaults for anything it leaves out."""
    return {**DEFAULTS, **(config.get("plots") or {})}


def display_name(model: str) -> str:
    """The model's name as written in the report, e.g. Diff-RNTraj for diffrntraj."""
    return NAMES.get(model, model)


def model_panels(result) -> tuple[Panel, list[Panel]]:
    """The real test panel and one panel per generated group of one converted model.

    A generated panel is titled with the model's name; the group is added in brackets only when the model has more than
    one, e.g. URBAN-SYN's finetune, scratch and frozen runs.
    """
    name = display_name(result.model)
    real = Panel("Real test", TrajSet.from_table(result.real), result.unit, real=True)
    groups = list(pd.unique(result.generated["group"]))
    generated = [Panel(name if len(groups) == 1 else f"{name} ({group})",
                       TrajSet.from_table(result.generated[result.generated["group"] == group]), result.unit, real=False)
                 for group in groups]
    return real, generated


def plot_trajectories(panels: list[Panel], grid: Grid, settings: dict, seed: int, title: str, path: Path) -> None:
    """Draw each panel's trajectories (all, or a random sample) as lines on the grid's extent."""
    from matplotlib.collections import LineCollection

    fig, axes = _figure(len(panels), grid, settings)
    for ax, panel in zip(axes, panels):
        chosen = sample_index(len(panel.trajectories), settings["sample"], seed)
        lines = [np.column_stack(panel.trajectories.points(k)[::-1]) for k in chosen]
        ax.add_collection(LineCollection(lines, colors=REAL_COLOR if panel.real else GENERATED_COLOR,
                                         linewidths=0.35, alpha=0.4))
        _style(ax, grid, f"{panel.title}\n{_count(len(chosen), len(panel.trajectories), panel.unit)}")
    fig.suptitle(title, fontsize=11)
    _save(fig, path, settings)


def plot_density(panels: list[Panel], grid: Grid, settings: dict, title: str, path: Path) -> None:
    """Draw the share of each panel's points in every grid cell, on one log color scale for all panels."""
    from matplotlib.colors import LogNorm

    shares, outside = [], []
    for panel in panels:
        cells = grid.cells(panel.trajectories.lat, panel.trajectories.lng)
        counts = np.bincount(cells[cells != OUTSIDE], minlength=grid.size).astype(float)
        shares.append(counts.reshape(grid.n_lat, grid.n_lng) / max(len(cells), 1))
        outside.append(int(np.sum(cells == OUTSIDE)))
    positive = np.concatenate([s[s > 0] for s in shares]) if any((s > 0).any() for s in shares) else np.array([1.0])
    norm = LogNorm(vmin=positive.min(), vmax=positive.max())

    fig, axes = _figure(len(panels), grid, settings, colorbar=True)
    image = None
    for ax, panel, share, out in zip(axes, panels, shares, outside):
        lng_edges = grid.lng_min + grid.cell_deg * np.arange(grid.n_lng + 1)
        lat_edges = grid.lat_min + grid.cell_deg * np.arange(grid.n_lat + 1)
        image = ax.pcolormesh(lng_edges, lat_edges, np.ma.masked_equal(share, 0), cmap="magma_r", norm=norm)
        note = f"{len(panel.trajectories.lat):,} points" + (f", {out:,} outside the grid" if out else "")
        _style(ax, grid, f"{panel.title}\n{note}")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.6, label=f"share of the panel's points in each {grid.cell_deg}° cell")
    fig.suptitle(title, fontsize=11)
    _save(fig, path, settings)


def main(argv=None) -> int:
    """Check the inputs, convert each requested model and write its trajectory and density plots."""
    args = parse_args(argv)
    try:
        import matplotlib
    except ImportError:
        print("ERROR  plotting needs matplotlib; in the dev container run: uv run python -m evaluation.plot ...",
              file=sys.stderr)
        return 1
    matplotlib.use("Agg")

    config = inputs.load_config(args.config)
    if "metrics" not in config:
        raise SystemExit("eval_config.yaml has no metrics: section")
    models = chosen_models(args.model, config)
    try:
        inputs.check_inputs(config, models, list(dict.fromkeys(model_network(m) for m in models)), fmm=False)
    except inputs.MissingInputsError as error:
        print(f"ERROR  {error}", file=sys.stderr)
        return 1
    print(f"All input files found for: {', '.join(models)}")

    settings = plot_settings(config)
    seed = int(config["metrics"]["seed"])
    grid_settings = config["metrics"]["grid"]
    grid = Grid.from_settings(grid_settings["cell_deg"], grid_settings["lng_bounds"], grid_settings["lat_bounds"])
    out = args.out / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_plots_" + "+".join(models))
    out.mkdir(parents=True, exist_ok=True)

    # the overview's one real test set: the model named in plots: overview_real if it is plotted, else the first one
    real_from = settings["overview_real"] if settings["overview_real"] in models else models[0]
    overview, overview_real = [], None
    for model in models:
        print(f"\n{model}: converting")
        result = convert_model(config, model, fmm=False, module="plot")
        real, generated = model_panels(result)
        name = display_name(model)
        plot_trajectories([real] + generated, grid, settings, seed, f"{name}: real test and generated trajectories",
                          out / f"{model}_trajectories.png")
        plot_density([real] + generated, grid, settings, f"{name}: share of points per grid cell",
                     out / f"{model}_density.png")
        print(f"  wrote {model}_trajectories.png and {model}_density.png")
        if model == real_from:
            overview_real = Panel(f"Real test ({name})", real.trajectories, real.unit, real=True)
        overview += generated

    if len(models) > 1:
        panels = [overview_real] + overview
        plot_trajectories(panels, grid, settings, seed, "Real test and generated trajectories",
                          out / "overview_trajectories.png")
        plot_density(panels, grid, settings, "Share of points per grid cell", out / "overview_density.png")
        print(f"\nwrote overview_trajectories.png and overview_density.png (real test set: {display_name(real_from)}'s)")
    print(f"Plots written to {out}")
    return 0


def _figure(n: int, grid: Grid, settings: dict, colorbar: bool = False):
    """A figure with n map panels, at most settings["columns"] per row, sized to the grid's shape."""
    import matplotlib.pyplot as plt

    columns = min(n, int(settings["columns"]))
    rows = -(-n // columns)
    height = PANEL_WIDTH_IN * _aspect(grid) * (grid.n_lat / grid.n_lng)
    fig, axes = plt.subplots(rows, columns, figsize=(PANEL_WIDTH_IN * columns + (1.2 if colorbar else 0),
                                                     height * rows + 0.6), squeeze=False, layout="constrained")
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, list(axes[:n])


def _style(ax, grid: Grid, title: str) -> None:
    """Fix the panel to the grid's extent with true proportions, and label it."""
    from matplotlib.ticker import MaxNLocator

    ax.set_xlim(grid.lng_min, grid.lng_min + grid.cell_deg * grid.n_lng)
    ax.set_ylim(grid.lat_min, grid.lat_min + grid.cell_deg * grid.n_lat)
    ax.set_aspect(_aspect(grid))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("longitude", fontsize=7)
    ax.set_ylabel("latitude", fontsize=7)


def _aspect(grid: Grid) -> float:
    """Degrees of latitude per degree of longitude on screen, so distances look the same in both directions."""
    middle = grid.lat_min + grid.cell_deg * grid.n_lat / 2
    return 1.0 / np.cos(np.radians(middle))


def _count(shown: int, total: int, unit: str) -> str:
    """'1,000 of 4,016 moves' or 'all 812 moves'."""
    unit = f" {unit}s" if unit else ""
    return f"{total:,}{unit}" if shown == total else f"{shown:,} of {total:,}{unit}"


def _save(fig, path: Path, settings: dict) -> None:
    """Write the figure as a PNG and free it."""
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=int(settings["dpi"]))
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
