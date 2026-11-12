"""Score baselines with the 12 metrics of the team's spec and write the results.

Run from the URBAN-SYN folder:
    python -m evaluation.evaluate --model diffrntraj
    python -m evaluation.evaluate --model tstrajgen mobilitygpt
    python -m evaluation.evaluate --model all
    python -m evaluation.evaluate --model difftraj --no-fmm   # GPS models without FMM: no road connectivity

DiffTraj and LSTM-TrajGAN get their road tables from FMM map matching, which runs first when the matches are missing or
out of date (fmm_prepare.py), so FMM must be installed, e.g. by running this in the fmm_scripts image.

Each run writes evaluation/results/<date>_<time>_<models>/ with scores.csv (every value, setting and sample size),
table.md (the report tables) and run.json (settings, inputs and diagnostics).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import fmm_prepare, inputs, tables
from evaluation.convert import chosen_models, convert_model, model_network
from evaluation.converters import CONVERTERS
from evaluation.metrics import METRICS
from evaluation.metrics.distribution import distribution_scores, grids_from_settings
from evaluation.metrics.privacy import privacy_scores
from evaluation.metrics.road import RoadSegments, off_network_scores, rsc_scores, successor_keys
from evaluation.metrics.trajectory import trajectory_scores
from evaluation.metrics.trajset import TrajSet

DEFAULT_OUT = Path(__file__).with_name("results")
SCORE_COLUMNS = ["model", "group", "metric", "setting", "pairing", "main_setting", "value", "std", "n", "note"]
TRAJECTORY_METRICS = ["hausdorff_km", "ndtw_km", "edr", "jaccard"]
SWEEP_SUMMARY = ["edr", "jaccard", "uniqueness", "off_network_rate"]  # the spec's "mean +- std: yes" metrics


def parse_args(argv=None) -> argparse.Namespace:
    """Read the command-line arguments."""
    parser = argparse.ArgumentParser(description="Score baselines with the common metrics.")
    parser.add_argument("--model", nargs="+", required=True,
                        help=f"model name(s) to evaluate, or 'all': {', '.join(CONVERTERS)}")
    parser.add_argument("--config", default=inputs.DEFAULT_CONFIG, help="path to eval_config.yaml")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="folder for the results (default: evaluation/results); each run gets a subfolder")
    parser.add_argument("--no-fmm", action="store_true",
                        help="don't map-match the GPS models (DiffTraj, LSTM-TrajGAN): no road connectivity for them")
    return parser.parse_args(argv)


def networks_needed(config: dict, models: list[str]) -> list[str]:
    """The canonical network for the off-network rate plus every network the models' road tables use."""
    needed = [config["metrics"]["off_network"]["network"]] + [model_network(m) for m in models]
    return list(dict.fromkeys(needed))


class Networks:
    """Loads each road network at most once per run."""

    def __init__(self, config: dict):
        """Remember the config; nothing is read until a network is needed."""
        self.config = config
        self._segments = None
        self._successors = {}

    def segments(self) -> RoadSegments:
        """The canonical network's road segments, for the off-network rate."""
        if self._segments is None:
            off = self.config["metrics"]["off_network"]
            if off["network"] != "tstrajgen":
                raise ValueError("The off-network rate can only read the tstrajgen network (nyc.geo) so far")
            reach = max(off["sweep_m"] + [off["tau_m"]])
            self._segments = RoadSegments.from_tstrajgen_geo(inputs.network_files(self.config, "tstrajgen")["geo"], reach)
        return self._segments

    def successors(self, network: str) -> np.ndarray:
        """The road-follows-road pairs of one network, for RSC."""
        if network not in self._successors:
            self._successors[network] = successor_keys(network, inputs.network_files(self.config, network))
        return self._successors[network]


def evaluate_model(config: dict, model: str, networks: Networks, log, fmm: bool = True, args: str = "",
                   result: tables.Converted | None = None) -> tuple[list[dict], dict]:
    """Convert one model, then score every generated group, the real-train floor and the real test set.

    Returns the score rows and a dict of notes, diagnostics and timings for run.json.
    """
    settings = config["metrics"]
    start = time.time()
    if result is None:
        result = convert_model(config, model, fmm=fmm, module="evaluate", args=args, log=log)
    problems = tables.validate(result)
    if problems:
        raise SystemExit(f"{model}: the converted tables failed their checks:\n  - " + "\n  - ".join(problems))
    if result.train is None:
        raise SystemExit(f"{model}: the converter returned no training set, which the privacy metrics and floor need")

    real = TrajSet.from_table(result.real)
    train = TrajSet.from_table(result.train)
    groups = list(pd.unique(result.generated["group"]))
    log(f"  read the data ({time.time() - start:.0f} s): {result.generated['traj_id'].nunique():,} generated "
        f"in {len(groups)} group(s), {len(real):,} real test and {len(train):,} real train trajectories")
    network = model_network(model)
    timings = {}
    rows, info = [], {"unit": result.unit, "native": result.native, "network": network,
                      "notes": {k: str(v) for k, v in result.notes.items()},
                      "sizes": {"real_test": len(real), "real_train": len(train)}, "groups": {}, "seconds": timings}

    for group in groups:
        step = time.time()
        gen = TrajSet.from_table(result.generated[result.generated["group"] == group])
        roads = None if result.generated_roads is None else \
            result.generated_roads[result.generated_roads["group"] == group]
        group_rows, diagnostics = score_set(gen, real, train, roads, result.native, network, networks, settings,
                                            timings.setdefault(group, {}))
        diagnostics["generated"] = len(gen)
        rows += _tag(group_rows, model, group)
        info["groups"][group] = diagnostics
        log(f"  scored {group} ({time.time() - step:.0f} s)")

    # Floor: real training trajectories in place of the generated set.
    step = time.time()
    floor_rows, floor_info = [], {}
    for name, scorer in (("distribution", distribution_scores), ("trajectory", trajectory_scores)):
        part_rows, part_info = _timed(timings.setdefault("real_train", {}), name, scorer, train, real, settings)
        if name == "trajectory":
            part_rows = [{**row, "pairing": part_info["pairing"]} for row in part_rows]
            # the floor scored with each comparison rule too, so every version B table has its own reference
            part_rows += compare_pairing_rows(train, real, settings, part_rows, part_info, floor_info,
                                              timings.setdefault("real_train", {}))
        floor_rows += part_rows
        floor_info.update(part_info)
    rows += _tag(floor_rows, model, "real_train")
    info["groups"]["real_train"] = floor_info

    # The real test set's own off-network rate and RSC, for reference.
    ref_time = timings.setdefault("real_test", {})
    ref_rows, ref_info = _timed(ref_time, "off_network", off_network_scores, networks.segments(), real, settings,
                                result.native)
    part_rows, part_info = _timed(ref_time, "rsc", rsc_scores, result.real_roads, networks.successors(network), len(real))
    rows += _tag(ref_rows + part_rows, model, "real_test")
    info["groups"]["real_test"] = {**ref_info, **part_info}
    log(f"  scored the real-data references ({time.time() - step:.0f} s)")
    info["total_seconds"] = round(time.time() - start, 1)
    return rows, info


def score_set(gen, real, train, roads, native, network, networks, settings, timings: dict) -> tuple[list[dict], dict]:
    """Score one generated group with all 12 metrics; returns rows and diagnostics (step times go into timings)."""
    rows, diagnostics = [], {}
    steps = [
        ("distribution", distribution_scores, (gen, real, settings)),
        ("trajectory", trajectory_scores, (gen, real, settings)),
        ("privacy", privacy_scores, (gen, train, real, settings)),
        ("off_network", off_network_scores, (networks.segments(), gen, settings, native)),
        ("rsc", rsc_scores, (roads, networks.successors(network), len(gen))),
    ]
    for name, scorer, args in steps:
        part_rows, part_info = _timed(timings, name, scorer, *args)
        if name == "trajectory":
            part_rows = [{**row, "pairing": part_info["pairing"]} for row in part_rows]
            part_rows += compare_pairing_rows(gen, real, settings, part_rows, part_info, diagnostics, timings)
        rows += part_rows
        diagnostics.update(part_info)
    return rows, diagnostics


def compare_rules(settings: dict) -> list[str]:
    """The rules in metrics.compare_pairing: one name (same_cell) or a list ([same_cell, nearest])."""
    rules = settings.get("compare_pairing") or []
    return [rules] if isinstance(rules, str) else list(rules)


def compare_pairing_rows(gen, real, settings, main_rows, main_info, diagnostics, timings) -> list[dict]:
    """Version B of the trajectory table: the trajectory metrics again under each rule of metrics.compare_pairing.

    One pairing rule for every model, so a model paired by ID can be compared like-for-like with Diff-RNTraj. Each rule
    gives its own rows (pairing "<rule> for all"); a set already paired by that rule (Diff-RNTraj and the floor with
    same_cell) repeats its main rows. None are main_setting, so the main and threshold tables stay on version A.
    """
    rows = []
    for rule in compare_rules(settings):
        if main_info["pairing"] == rule:
            rule_rows, info = main_rows, main_info
        else:
            without_ids = TrajSet(gen.ids, np.full(len(gen), None, dtype=object), gen.offsets, gen.lat, gen.lng)
            rule_settings = {**settings, "fallback_pairing": {**settings["fallback_pairing"], "rule": rule}}
            rule_rows, info = _timed(timings, f"trajectory ({rule})", trajectory_scores, without_ids, real,
                                     rule_settings)
        diagnostics[f"compare_pairing_{rule}"] = info
        rows += [{**row, "pairing": f"{rule} for all", "main_setting": False} for row in rule_rows]
    return rows


def add_sweep_summaries(scores: pd.DataFrame) -> pd.DataFrame:
    """Add one "mean+-std" row per model, group, pairing and sweep metric: mean and std (K - 1) over its settings."""
    extra = []
    sweeps = scores[scores["metric"].isin(SWEEP_SUMMARY)].assign(pairing=lambda d: d["pairing"].fillna(""))
    for (model, group, pairing, metric), part in sweeps.groupby(["model", "group", "pairing", "metric"], sort=False):
        values = part["value"].to_numpy(dtype=float)
        extra.append({"model": model, "group": group, "metric": metric, "setting": "mean+-std", "pairing": pairing,
                      "main_setting": False, "value": float(np.mean(values)),
                      "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                      "n": int(part["n"].iloc[0]), "note": "over " + ", ".join(part["setting"])})
    return pd.concat([scores, pd.DataFrame(extra, columns=SCORE_COLUMNS)], ignore_index=True)


# The three results tables of table.md, in the team's report layout: (title, [(metric, column header)]).
REPORT_TABLES = [
    ("Distribution Similarity", [("geo_jsd", "Global geographic ↓"), ("od_jsd", "Start/End Point ↓"),
                                 ("start_jsd", "Start point ↓"), ("end_jsd", "End point ↓"),
                                 ("distance_jsd", "Trip Distance ↓"), ("radius_jsd", "Radius of gyration ↓")]),
    ("Trajectory-Level Similarity", [("hausdorff_km", "Hausdorff (km) ↓"), ("ndtw_km", "DTW (km) ↓"),
                                     ("edr", "EDR ↓"), ("jaccard", "Jaccard ↑")]),
    ("Privacy and Road Validity", [("mia_auc", "MIA (0.5 best)"), ("uniqueness", "Uniqueness ↑"),
                                   ("off_network_rate", "Off-network rate ↓"),
                                   ("rsc", "Snapped connectivity ↑")]),
]
# Columns of the real-data reference table (real training data for all but the last two, the real test set for those).
REFERENCE_COLUMNS = [("geo_jsd", "Global geographic"), ("od_jsd", "Start/End Point"), ("start_jsd", "Start point"),
                     ("end_jsd", "End point"), ("distance_jsd", "Trip Distance"),
                     ("radius_jsd", "Radius of gyration"), ("hausdorff_km", "Hausdorff (km)"), ("ndtw_km", "DTW (km)"),
                     ("edr", "EDR"), ("jaccard", "Jaccard"), ("off_network_rate", "Off-network rate"),
                     ("rsc", "Snapped connectivity")]


def report_rows(scores: pd.DataFrame, config: dict) -> list[tuple[str, str, str, str]]:
    """The table rows as (model, group, label, name), ordered and named by the report: section of the config.

    label comes from report.group_labels (empty for a model with one unlabelled group, the group name otherwise); name
    is the model's display name.
    """
    report = config.get("report") or {}
    names, labels = report.get("names") or {}, report.get("group_labels") or {}
    pairs = [(m, g) for m, g in dict.fromkeys(zip(scores["model"], scores["group"])) if not g.startswith("real_")]
    present = list(dict.fromkeys(m for m, _ in pairs))
    order = [m for m in report.get("order") or [] if m in present] + [m for m in present if m not in (report.get("order") or [])]
    rows = []
    for model in order:
        groups = [g for m, g in pairs if m == model]
        for group in groups:
            label = (labels.get(model) or {}).get(group)
            if label is None:
                label = "" if len(groups) == 1 else group
            rows.append((model, group, label, names.get(model, model)))
    return rows


def make_table(scores: pd.DataFrame, config: dict, run_id: str) -> str:
    """Write table.md: the three report tables, the real-data reference and the other settings."""
    canonical = scores[scores["main_setting"].astype(bool)]
    value = {(m, g, k): v for m, g, k, v in zip(canonical["model"], canonical["group"], canonical["metric"],
                                                canonical["value"])}
    # Version B of the trajectory metrics (one pairing rule for every model), at the canonical settings
    main_setting = dict(zip(canonical["metric"], canonical["setting"]))
    version_b = scores[scores["pairing"].fillna("").str.endswith(" for all")]
    version_b = version_b[np.array([main_setting.get(m) == t for m, t in zip(version_b["metric"], version_b["setting"])],
                                   dtype=bool)]  # a bool array: an empty list would select columns instead
    value_b = {(p, m, g, k): v for p, m, g, k, v in zip(version_b["pairing"], version_b["model"], version_b["group"],
                                                        version_b["metric"], version_b["value"])}
    note_b = {(p, m, g, k): n for p, m, g, k, n in zip(version_b["pairing"], version_b["model"], version_b["group"],
                                                       version_b["metric"], version_b["note"])}
    note = {(m, g, k): n for m, g, k, n in zip(canonical["model"], canonical["group"], canonical["metric"],
                                               canonical["note"])}
    rows = report_rows(scores, config)
    lines = ["# Evaluation results", ""]
    def score(model, group, metric) -> str:
        """One cell; snapped connectivity also shows the share FMM matched when it is not the whole set."""
        text = _cell(value.get((model, group, metric)))
        return text + _matched(note.get((model, group, metric))) if text and metric == "rsc" else text

    for title, metrics in REPORT_TABLES:
        trajectory = title.startswith("Trajectory")  # gets a Paired column: how the pairs were made, what share has one
        lines += [f"## {title}", "",
                  "| Model | | " + ("Paired | " if trajectory else "") + " | ".join(h for _, h in metrics) + " |",
                  "|---|---|" + ("---|" if trajectory else "") + "---|" * len(metrics)]
        for model, group, label, name in rows:
            paired = [_pairing(note.get((model, group, "hausdorff_km"))).removeprefix("by ")] if trajectory else []
            cells = paired + [score(model, group, metric) for metric, _ in metrics]
            lines.append(f"| {label} | {name} | " + " | ".join(cells) + " |")
        lines.append("")
        present = set(version_b["pairing"])
        order = [f"{r} for all" for r in compare_rules(config["metrics"])]  # the order of compare_pairing in the config
        for pairing in ([p for p in order if p in present] if title.startswith("Trajectory") else []):
            lines += ["| Model | | Paired | " + " | ".join(h for _, h in metrics) + " |",
                      "|---|---|---|" + "---|" * len(metrics)]
            for model, group, label, name in rows:
                paired = _pairing(note_b.get((pairing, model, group, "hausdorff_km"))).removeprefix("by ")
                cells = [_cell(value_b.get((pairing, model, group, metric))) for metric, _ in metrics]
                lines.append(f"| {label} | {name} | {paired} | " + " | ".join(cells) + " |")
            for model in dict.fromkeys(model for model, *_ in rows):
                if (pairing, model, "real_train", "hausdorff_km") in value_b:
                    name = next(n for m, _, _, n in rows if m == model)
                    paired = _pairing(note_b.get((pairing, model, "real_train", "hausdorff_km"))).removeprefix("by ")
                    cells = [_cell(value_b.get((pairing, model, "real_train", metric))) for metric, _ in metrics]
                    lines.append(f"| real data (reference) | {name} | {paired} | " + " | ".join(cells) + " |")
            lines.append("")

    models = list(dict.fromkeys(model for model, *_ in rows))
    names = {model: name for model, _, _, name in rows}
    lines += ["## Real data scored the same way", "",
              "The reference for each model: real training data scored in place of the generated set (all but the "
              "last two columns), and the real test set itself (last two). It is roughly the best score a model can get.", "",
              "| Real data for | " + " | ".join(h for _, h in REFERENCE_COLUMNS) + " |",
              "|---|" + "---|" * len(REFERENCE_COLUMNS)]
    for model in models:
        cells = [score(model, "real_train" if i < len(REFERENCE_COLUMNS) - 2 else "real_test", metric)
                 for i, (metric, _) in enumerate(REFERENCE_COLUMNS)]
        lines.append(f"| {names[model]} | " + " | ".join(cells) + " |")
    lines.append("")

    columns = [(model, group) for model, group, _, _ in rows]
    header = ["| Metric | " + " | ".join(f"{name} ({label})" if label else name for _, _, label, name in rows) + " |",
              "|---|" + "---|" * len(rows)]
    lines += _settings_table(scores, columns, header)
    return "\n".join(lines)


# Metrics that depend on a threshold or grid size, shown at every setting in the last table of table.md.
SETTING_ROWS = [
    ("edr", "EDR, ε ="),
    ("jaccard", "Jaccard, grid"),
    ("uniqueness", "Uniqueness, threshold"),
    ("off_network_rate", "Off-network rate, threshold"),
    ("geo_jsd", "Global geographic, grid"),
    ("od_jsd", "Start/End Point, grid"),
    ("start_jsd", "Start point, grid"),
    ("end_jsd", "End point, grid"),
]


def _settings_table(scores: pd.DataFrame, columns: list, header: list[str]) -> list[str]:
    """The last table of table.md: each threshold-dependent metric at each of the spec's settings."""
    rows = scores[scores["metric"].isin([m for m, _ in SETTING_ROWS]) & (scores["setting"] != "mean+-std")
                  & ~scores["pairing"].fillna("").str.endswith(" for all")]  # version A only
    if rows.empty:
        return []
    value = {(m, g, k, t): v for m, g, k, t, v in zip(rows["model"], rows["group"], rows["metric"],
                                                      rows["setting"], rows["value"])}
    lines = ["## The same metrics at the spec's other settings", "",
             "These metrics depend on a threshold or grid size. The main setting (used in the tables above) "
             "is marked; the other two show how much the score depends on that choice.", ""]
    lines += header
    for metric, label in SETTING_ROWS:
        part = rows[rows["metric"] == metric]
        for setting in sorted(part["setting"].unique(), key=_setting_number):
            main = bool(part.loc[part["setting"] == setting, "main_setting"].astype(bool).any())
            cells = [_cell(value.get((model, group, metric, setting))) for model, group in columns]
            shown = setting.split("=")[1].replace("m", " m") if metric in ("edr", "uniqueness", "off_network_rate") \
                else setting.split("=")[1] + "°"
            lines.append(f"| {label} {shown}{' (main)' if main else ''} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _setting_number(setting: str) -> float:
    """The number in a setting label such as eps=100m or grid=0.005, for sorting."""
    return float(setting.split("=")[1].rstrip("m"))


def _cell(value) -> str:
    """Format one value with 4 decimals (rates as fractions, as in the report tables)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{value:.4f}"


def _pairing(note) -> str:
    """How a row's trajectory metrics were paired, from the score note."""
    if not note:
        return ""
    if str(note).startswith("id"):
        return "by ID"
    share = str(note).split(", ")[-1].split("%")[0]
    rule = "same start and end cell" if str(note).startswith("same_cell") else "nearest real trajectory"
    return f"by {rule}, {share}% of the generated set has a match"


def _coverage(note) -> str:
    """The share of trajectories with a road sequence for snapped connectivity, from the score note."""
    if not note:
        return ""
    with_roads, total = (int(n.replace(",", "")) for n in str(note).split(" trajectories")[0].split(" of "))
    share = 100 * with_roads / total
    return "all" if with_roads == total else f"{share:.0f}%" if 1 <= share < 99.5 else f"{share:.1f}%"


def _matched(note) -> str:
    """' (47% matched)' when FMM gave roads to only part of the set (from the snapped connectivity note), else ''."""
    share = _coverage(note)
    return f" ({share} matched)" if share not in ("", "all") else ""


def run_info(config: dict, models: list[str], networks: list[str]) -> dict:
    """Settings and input files of this run, for run.json."""
    root = config["repo_root"]

    def describe(path: Path) -> dict:
        """Path (relative to the repo), size and modification time of one input file or folder."""
        stat = path.stat()
        shown = path.relative_to(root) if path.is_relative_to(root) else path
        return {"path": shown.as_posix(), "bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")}

    return {
        "config": str(config["config_path"]),
        "git_commit": _git_commit(root),
        "metrics": config["metrics"],
        "grids": {str(size): f"{g.n_lng} x {g.n_lat}" for size, g in grids_from_settings(config["metrics"]).items()},
        "inputs": {m: {k: describe(p) for k, p in inputs.model_files(config, m).items()} for m in models},
        "networks": {n: {k: describe(p) for k, p in inputs.network_files(config, n).items()} for n in networks},
    }


def main(argv=None) -> int:
    """Check the inputs, score each requested model and write scores.csv, table.md and run.json."""
    args = parse_args(argv)
    config = inputs.load_config(args.config)
    if "metrics" not in config:
        raise SystemExit("eval_config.yaml has no metrics: section")
    models = chosen_models(args.model, config)
    needed = networks_needed(config, models)
    try:
        inputs.check_inputs(config, models, needed, fmm=not args.no_fmm)
    except inputs.MissingInputsError as error:
        print(f"ERROR  {error}", file=sys.stderr)
        return 1
    print(f"All input files found for: {', '.join(models)}")

    started = datetime.now()
    run_id = started.strftime("%Y-%m-%d_%H%M%S") + "_" + "+".join(models)
    info = {"run_id": run_id, "started": started.isoformat(timespec="seconds"), **run_info(config, models, needed),
            "models": {}}
    networks = Networks(config)
    command = fmm_prepare.command_args(argv)
    # Convert (and map-match) the GPS models before scoring anything, so a missing FMM program stops the run at the
    # start, not after the other models.
    converted = {}
    if not args.no_fmm:
        for model in [m for m in models if fmm_prepare.uses_fmm(config, m)]:
            print(f"\n{model}: converting and checking its FMM matches")
            converted[model] = convert_model(config, model, module="evaluate", args=command)
    rows = []
    for model in models:
        print(f"\n{model}")
        model_rows, info["models"][model] = evaluate_model(config, model, networks, log=print, fmm=not args.no_fmm,
                                                           args=command, result=converted.pop(model, None))
        rows += model_rows

    scores = add_sweep_summaries(pd.DataFrame(rows, columns=SCORE_COLUMNS))
    out = args.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False)
    table = make_table(scores, config, run_id)
    (out / "table.md").write_text(table, encoding="utf-8")
    info["finished"] = datetime.now().isoformat(timespec="seconds")
    (out / "run.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print("\n" + table)
    print(f"Results written to {out}")
    return 0


def _timed(timings: dict, name: str, scorer, *args):
    """Run one scorer and record how long it took (seconds) under timings[name]."""
    start = time.time()
    result = scorer(*args)
    timings[name] = round(time.time() - start, 1)
    return result


def _tag(rows: list[dict], model: str, group: str) -> list[dict]:
    """Add the model and group to every score row."""
    return [{"model": model, "group": group, **row} for row in rows]


def _git_commit(root: Path) -> str | None:
    """The repo's current commit (with "+changes" when there are uncommitted changes), or None without git."""
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("+changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    sys.exit(main())
