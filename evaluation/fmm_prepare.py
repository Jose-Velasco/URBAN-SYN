"""Map-match the GPS models' trajectories with FMM as part of the evaluation.

DiffTraj and LSTM-TrajGAN write GPS points, so their road tables (for road connectivity) come from FMM map matching onto
TS-TrajGen's road network. add_road_tables() makes the matches the first time a model is converted and again whenever
its trajectories, the road network or the FMM settings change; otherwise it reuses the last run's matches.

For each set of a model (generated, real), these files are kept in the model's fmm_dir (eval_config.yaml):
  <model>_<set>_fmm_points.csv       FMM's input: id;x;y;timestamp (timestamp = point order)
  <model>_<set>_fmm_trip_id_map.csv  FMM's integer id -> the converted table's traj_id
  <model>_<set>_fmm_config.xml       the FMM config: the template with these files filled in
  <model>_<set>_fmm_match.csv        FMM's output: id;cpath;opath;error
  <model>_<set>_fmm_source.json      fingerprint of the inputs the match was made from

The fingerprint stops an old match from being used for new trajectories: DiffTraj and LSTM-TrajGAN reuse their
trajectory IDs from run to run, so a stale match would attach its routes to the wrong trajectories.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import inputs
from evaluation.fmm_matches import read_routes
from evaluation.tables import Converted, road_table

# converted table (also the file name part) -> road table attribute
SETS = {"generated": "generated_roads", "real": "real_roads"}
OUTPUT_FIELDS = ("cpath", "opath", "error")  # what FMM writes; the converters use cpath or opath

DOCKER_HINT = (
    "Run the evaluation in the fmm_scripts image, which has FMM, from the URBAN-SYN folder\n"
    "(build it once with: docker compose -f fmm_scripts/docker-compose.yml build):\n"
    "  docker compose -f fmm_scripts/docker-compose.yml run --rm -T -w /urban-syn fmm "
    "python -m evaluation.{module} {args}\n"
    "or add --no-fmm to score these models without road connectivity."
)

_checked_networks: set[str] = set()


class FmmUnavailableError(SystemExit):
    """FMM has to run but its program is not installed where the evaluation runs."""


def uses_fmm(config: dict, model: str) -> bool:
    """True for the models whose road tables come from FMM (an fmm_dir in eval_config.yaml)."""
    return "fmm_dir" in config["models"][model]


def add_road_tables(config: dict, model: str, result: Converted, network: str, module: str = "evaluate",
                    args: str = "", log=print) -> None:
    """Fill result.generated_roads and result.real_roads from FMM matches, making the matches when needed."""
    settings = inputs.fmm_settings(config)
    if network == "tstrajgen":
        _check_same_network(config, settings)
    folder = inputs.resolve(config, config["models"][model]["fmm_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    route = inputs.model_options(config, model).get("fmm_route", "cpath")
    for name, attribute in SETS.items():
        table = getattr(result, name)
        stem = folder / f"{model}_{name}_fmm"
        routes, made = _routes(table, stem, settings, route, f"{model} ({name})", module, args, log)
        groups = table.drop_duplicates("traj_id").set_index("traj_id")["group"]
        setattr(result, attribute, road_table(
            ((traj_id, groups[traj_id], routes[traj_id]) for traj_id in groups.index if traj_id in routes), network))
        result.notes[f"FMM matched ({name})"] = f"{len(routes):,} of {len(groups):,}" + (" (new match)" if made else "")
    result.notes["FMM route"] = route


def _routes(table: pd.DataFrame, stem: Path, settings: dict, route: str, label: str, module: str, args: str,
            log) -> tuple[dict, bool]:
    """Return ({traj_id: road IDs}, whether FMM ran) for one GPS table, running FMM if the match is missing or old."""
    paths = {part: Path(f"{stem}_{part}") for part in
             ("points.csv", "trip_id_map.csv", "config.xml", "match.csv", "source.json")}
    points, id_map = _fmm_input(table)
    points_csv = points.to_csv(sep=";", index=False, float_format="%.8f", lineterminator="\n")
    config_xml = _config_xml(settings, paths["points.csv"], paths["match.csv"])
    fingerprint = {
        "points_sha256": hashlib.sha256(points_csv.encode()).hexdigest(),
        "config_sha256": hashlib.sha256(config_xml.encode()).hexdigest(),
        "network": _file_stamp(settings["network"]),
        "ubodt": _file_stamp(settings["ubodt"]),
    }
    if _up_to_date(paths, fingerprint):
        # Same points in the same order, so FMM's id i is still the i-th trajectory. The ID map is rewritten from the
        # current table in case trajectories were renamed since: the fingerprint covers points, not traj_ids.
        id_map.to_csv(paths["trip_id_map.csv"], index=False)
        return read_routes(paths["match.csv"], paths["trip_id_map.csv"], route), False

    command = shlex.split(settings["command"])
    if shutil.which(command[0]) is None:
        raise FmmUnavailableError(
            f"ERROR  The FMM match for {label} is missing or older than its trajectories, and FMM's "
            f"'{command[0]}' program is not available here.\n" + DOCKER_HINT.format(module=module, args=args))

    paths["source.json"].unlink(missing_ok=True)       # a failed run never leaves a valid-looking match behind
    paths["points.csv"].write_text(points_csv, encoding="utf-8")
    id_map.to_csv(paths["trip_id_map.csv"], index=False)
    paths["config.xml"].write_text(config_xml, encoding="utf-8")
    log(f"  map-matching {label}: {len(id_map):,} trajectories, {len(points):,} points ...")
    done = subprocess.run(command + [os.path.relpath(paths["config.xml"], settings["workdir"])],
                          cwd=settings["workdir"], capture_output=True, text=True)
    output = (done.stdout + done.stderr).strip().splitlines()
    if done.returncode != 0 or not paths["match.csv"].exists():
        tail = "\n    ".join(output[-15:])
        raise SystemExit(f"ERROR  FMM failed for {label} (exit code {done.returncode}):\n    {tail}")
    for line in output:
        if "total points" in line:
            log("  " + line.split("]", 1)[-1].strip())
    paths["source.json"].write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    return read_routes(paths["match.csv"], paths["trip_id_map.csv"], route), True


def _fmm_input(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FMM's point table (id;x;y;timestamp, sorted by id and point order) and the id -> traj_id map.

    FMM needs integer ids, so trajectories are numbered 0, 1, 2... in order of first appearance; the timestamp is the
    point's position in its trajectory (seq), which is all FMM needs for geometric matching.
    """
    codes, traj_ids = pd.factorize(table["traj_id"], sort=False)
    seq = table["seq"].to_numpy()
    order = np.lexsort((seq, codes))
    points = pd.DataFrame({
        "id": codes[order],
        "x": table["lng"].to_numpy(float)[order],
        "y": table["lat"].to_numpy(float)[order],
        "timestamp": seq[order],
    })
    id_map = pd.DataFrame({"source_traj_id": traj_ids.astype(str), "id": np.arange(len(traj_ids))})
    return points, id_map


def _config_xml(settings: dict, points: Path, match: Path) -> str:
    """The template config with the GPS file, output file and output fields replaced.

    Paths are relative to workdir, so the config (and fingerprint) is the same in the dev container and Docker image.
    """
    tree = ET.ElementTree(ET.fromstring(settings["template_text"]))
    root = tree.getroot()
    root.find("input/gps/file").text = os.path.relpath(points, settings["workdir"]).replace(os.sep, "/")
    output = root.find("output")
    output.find("file").text = os.path.relpath(match, settings["workdir"]).replace(os.sep, "/")
    fields = output.find("fields")
    for child in list(fields):
        fields.remove(child)
    for name in OUTPUT_FIELDS:
        ET.SubElement(fields, name)
    ET.indent(tree)
    return ET.tostring(root, encoding="unicode") + "\n"


def _up_to_date(paths: dict, fingerprint: dict) -> bool:
    """True when the match and its ID map exist and were made from exactly these inputs."""
    if not all(paths[p].exists() for p in ("match.csv", "trip_id_map.csv", "source.json")):
        return False
    try:
        return json.loads(paths["source.json"].read_text(encoding="utf-8")) == fingerprint
    except (OSError, ValueError):
        return False


def _file_stamp(path: Path) -> dict:
    """Size and whole-second modification time of a file, enough to notice that it was rebuilt."""
    stat = path.stat()
    return {"file": path.name, "bytes": stat.st_size, "modified": int(stat.st_mtime)}


def _check_same_network(config: dict, settings: dict) -> None:
    """Stop if FMM's network and TS-TrajGen's nyc.geo have a different number of roads.

    The road tables are scored on TS-TrajGen's network, which only works when FMM's road IDs are TS-TrajGen's geo_ids.
    Both number the roads 0, 1, 2... from the same nyc.gpkg, so a different count means they come from different
    downloads.
    """
    geo = inputs.network_files(config, "tstrajgen")["geo"]
    key = f"{geo}|{settings['network']}"
    if key in _checked_networks:
        return
    geo_roads = len(pd.read_csv(geo, usecols=[0]))
    dbf = settings["network"].with_suffix(".dbf")
    with dbf.open("rb") as handle:
        fmm_roads = int.from_bytes(handle.read(8)[4:8], "little")
    if geo_roads != fmm_roads:
        raise SystemExit(
            f"ERROR  FMM's road network ({settings['network'].name}: {fmm_roads:,} roads) and TS-TrajGen's "
            f"nyc.geo ({geo_roads:,} roads) differ, so FMM's road IDs would not be TS-TrajGen's geo_ids. "
            "Use the network that TS-TrajGen was built on (fmm_scripts/prepare_fmm_inputs.py writes both).")
    _checked_networks.add(key)


def command_args(argv: list[str] | None) -> str:
    """The command-line arguments of this run, for the Docker hint."""
    return " ".join(shlex.quote(a) for a in (sys.argv[1:] if argv is None else argv))
