#!/usr/bin/env python3
"""
Run the whole NYC Diff-RNTraj data pipeline with one command.

From PowerShell (or any terminal on your computer), in URBAN-SYN/fmm_scripts/diffrntraj_version/:

    python run_diffrntraj_pipeline.py

What it runs, in order. Every step runs in this folder's Docker image (Dockerfile and
docker-compose.yml here), so nothing has to be installed on your computer except Docker Desktop.
The result is everything training needs, in baselines/diffusion/diff_rntraj/repo/data/NYC/:

  step      runs                      what
  --------  ------------------------  ----------------------------------------------------
  prep      python                    prepare_fmm_inputs_diffrntraj.py: download road network,
                                      build the 15 s GPS points CSV from the parquet
                                      (needs internet for the OpenStreetMap download)
  ubodt     FMM                       ubodt_gen  (shortest-path table)
  fmm       FMM                       fmm        (map matching)
  convert   python                    convert_fmm_to_diffrntraj.py -> baselines/.../repo/data/NYC/
                                      (writes straight into the model's data folder; first clears
                                      the old graph files, which no longer match the new road IDs)
  graph     python                    build_graph_nyc.py (graph.pkl, graph_X.csv, edge list)
  embed     python (GPU if any)       train_road_embed.py (road_embed.txt) - can be slow,
                                      only runs if you ask for it with --to embed

Optional:
    python run_diffrntraj_pipeline.py --from fmm        # resume from a step
    python run_diffrntraj_pipeline.py --only convert    # run one step
    python run_diffrntraj_pipeline.py --to embed        # also train the road embedding
    python run_diffrntraj_pipeline.py --dry-run         # print the commands only
    python run_diffrntraj_pipeline.py --to embed --embed-cpu   # road embedding without an NVIDIA GPU (slow)

Image: build it once with `docker compose build` in this folder (the first run also builds it if
it's missing; it compiles FMM, so it takes a while). It mounts the URBAN-SYN folder at
/workspace/URBAN-SYN. The embed step uses the same image with the GPU (service diffrntraj-fmm-gpu).
Training and generation then run in the model image (baselines/diffusion/diff_rntraj/), on the HPC.

File paths below match config/*_diffrntraj.xml. The FMM-related paths are read
from those XML files, so the two can't drift apart.

Folder layout: everything for Diff-RNTraj is in fmm_scripts/diffrntraj_version/ (scripts,
configs, Docker files, data/, logs/). The TS-TrajGen files stay in fmm_scripts/; this pipeline
only uses their utils.py and config/nyc.yaml. FMM is built into this folder's image the same way
as in fmm_scripts/dockerfile, so the TS-TrajGen containers aren't needed here (they still run
the TS-TrajGen pipeline).
"""

# Works with Python 3.6+ (only the standard library).
import argparse
import codecs
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Paths (relative to fmm_scripts/diffrntraj_version/). Change here if you rename files.
HERE = Path(__file__).resolve().parent                     # URBAN-SYN/fmm_scripts/diffrntraj_version
REPO = HERE.parents[1]                                     # URBAN-SYN

# Road network settings: the shared TS-TrajGen config/nyc.yaml (drive-only roads, trajectory bounding box).
ROAD_CONFIG = "../config/nyc.yaml"
UBODT_XML = "./config/ubodt_config_diffrntraj.xml"
FMM_XML = "./config/fmm_config_csv_point_diffrntraj.xml"

PARQUET = "../../data/nyc_output_tabular/output/enriched_moves.parquet"
CANONICAL_GPKG = "./data/road_network/nyc_drive.gpkg"
TRIP_ID_MAP = "./data/nyc_gps_points_fmm_trip_id_map_drive.csv"
# uid, tid of TS-TrajGen's test trips (held out in gen_test/ by the convert step)
TEST_TRIPS = "../nyc_test.csv"                                   # in fmm_scripts/
PREP_LOG = "./logs/prepare_fmm_nyc_diffrntraj.log"
# Everything the pipeline prints (all steps), appended run after run.
PIPELINE_LOG = "./logs/run_diffrntraj_pipeline.log"
# The converter writes straight into the model's data folder (no copy step).
# Relative to fmm_scripts/diffrntraj_version/, so it works on this computer and in the container.
CONVERTER_OUT = "../../baselines/diffusion/diff_rntraj/repo/data/NYC"

MODEL_REPO = REPO / "baselines/diffusion/diff_rntraj/repo"
MODEL_DATA = MODEL_REPO / "data/NYC"

# Containers. URBAN-SYN is mounted at ROOT_IN_CONTAINER in both, so the relative paths above
# work the same inside them.
FMM_COMPOSE = HERE / "docker-compose.yml"                  # this folder's image
FMM_SERVICE = "diffrntraj-fmm"                             # no GPU
GPU_SERVICE = "diffrntraj-fmm-gpu"                         # same image with the GPU (embed step)
FMM_IMAGE = "diffrntraj-fmm:latest"
EMBED_CPU = False                                          # --embed-cpu
ROOT_IN_CONTAINER = "/workspace/URBAN-SYN"
PIPELINE_DIR_IN_CONTAINER = f"{ROOT_IN_CONTAINER}/fmm_scripts/diffrntraj_version"
MODEL_DIR_IN_CONTAINER = f"{ROOT_IN_CONTAINER}/baselines/diffusion/diff_rntraj/repo"

STEPS = ["prep", "ubodt", "fmm", "convert", "graph", "embed"]
DEFAULT_LAST = "graph"


# Helpers
DRY_RUN = False


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def banner(text: str) -> None:
    """Framed heading, used once at the start of a run."""
    print("\n" + "=" * 78 + f"\n  {text}\n  {now()}\n" + "=" * 78, flush=True)


def heading(text: str) -> None:
    """Plain heading for each step, labelled with the date and time."""
    print(f"\n[{now()}] {text}", flush=True)


def run(cmd, cwd=HERE):
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"$ {printable}", flush=True)
    if DRY_RUN:
        return
    # Stream the output through sys.stdout, so it is shown live and also saved to the log.
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            # PYTHONIOENCODING: Windows would otherwise print through cp1252 and fail on non-ASCII text
                            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        chunk = proc.stdout.read1(4096)
        if not chunk:
            break
        sys.stdout.write(decoder.decode(chunk))
        sys.stdout.flush()
    sys.stdout.write(decoder.decode(b"", final=True))
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"command failed (exit code {returncode})")


class TeeToLog:
    """Print to the console and append the same text to PIPELINE_LOG.

    Progress bars redraw a line with carriage returns; they still animate on
    screen, but only the final state of each line is written to the file.
    """

    def __init__(self, console, path: Path) -> None:
        self.console = console
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")
        self.pending = ""

    def write(self, text: str) -> int:
        self.console.write(text)
        self.pending += text
        *lines, self.pending = self.pending.split("\n")
        for line in lines:
            parts = [part for part in line.split("\r") if part.strip()]
            self.file.write((parts[-1] if parts else "") + "\n")
        self.file.flush()
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def start_pipeline_log() -> None:
    if isinstance(sys.stdout, TeeToLog):  # already logging
        return
    path = HERE / PIPELINE_LOG
    new_file = not path.exists() or path.stat().st_size == 0
    sys.stdout = sys.stderr = TeeToLog(sys.stdout, path)
    # The run's banner (with date and time) starts each run; a blank line keeps runs apart.
    if not new_file:
        sys.stdout.file.write("\n")


def xml_files(path):
    """All <file> entries in an FMM XML config, ignoring commented-out ones."""
    text = (HERE / path).read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return re.findall(r"<file>\s*(.*?)\s*</file>", text)


def in_container(compose: Path, service: str, workdir: str, *cmd: str) -> None:
    """Run one command in a container (docker compose run of the given compose file)."""
    run(["docker", "compose", "-f", str(compose), "run", "--rm", "-T", "-e", "PYTHONUNBUFFERED=1",
         "-w", workdir, service, *cmd])


def in_pipeline_dir(*cmd: str) -> None:
    """FMM pipeline image, in fmm_scripts/diffrntraj_version/."""
    in_container(FMM_COMPOSE, FMM_SERVICE, PIPELINE_DIR_IN_CONTAINER, *cmd)


def in_model_dir(*cmd: str, gpu: bool = False) -> None:
    """Same image, in the model's repo/ folder (build_graph_nyc.py and train_road_embed.py live there)."""
    in_container(FMM_COMPOSE, GPU_SERVICE if gpu else FMM_SERVICE, MODEL_DIR_IN_CONTAINER, *cmd)


def docker_image_exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


# Steps
UBODT_FILES = xml_files(UBODT_XML)          # [network shp, ubodt output]
FMM_FILES = xml_files(FMM_XML)              # [ubodt, network shp, gps csv, match csv]
FMM_NETWORK, UBODT_BIN = UBODT_FILES[0], UBODT_FILES[1]
GPS_CSV, MATCH_CSV = FMM_FILES[2], FMM_FILES[3]


def step_prep() -> None:
    # Each run is appended to the same log; a blank line keeps the runs apart.
    log = HERE / PREP_LOG
    if not DRY_RUN and log.exists() and log.stat().st_size > 0:
        with log.open("a", encoding="utf-8") as f:
            f.write("\n")
    in_pipeline_dir(
        "python", "prepare_fmm_inputs_diffrntraj.py",
        "--config", ROAD_CONFIG,
        "--canonical_road_network_output", CANONICAL_GPKG,
        "--canonical_road_network_layer", "roads",
        "--fmm_road_network_output", FMM_NETWORK,
        "--trajectory_parquet", PARQUET,
        "--gps_output", GPS_CSV,
        "--trip_id_map_output", TRIP_ID_MAP,
        "--log_file", PREP_LOG,
    )


def step_ubodt() -> None:
    in_pipeline_dir("ubodt_gen", UBODT_XML)


def step_fmm() -> None:
    in_pipeline_dir("mkdir", "-p", str(Path(MATCH_CSV).parent.as_posix()))
    in_pipeline_dir("fmm", FMM_XML)


def step_convert() -> None:
    clear_stale_graph_files()
    in_pipeline_dir(
        "python", "convert_fmm_to_diffrntraj.py",
        "--fmm-output", MATCH_CSV,
        "--road-network", CANONICAL_GPKG,
        "--output-dir", CONVERTER_OUT,
        # Change for Diff-RNTraj (NYC): hold out TS-TrajGen's test trips in gen_test/
        "--trip-map", TRIP_ID_MAP,
        "--test-trips", TEST_TRIPS,
    )


def clear_stale_graph_files() -> None:
    # Road IDs change whenever the data is converted again, so graph files built
    # from the previous IDs are wrong now. They are all rebuilt by later steps.
    graph = MODEL_DATA / "graph"
    print(f"clear old graph files in {graph}")
    if DRY_RUN:
        return
    graph.mkdir(parents=True, exist_ok=True)
    for name in ("graph.pkl", "graph_X.csv", "graph_edge.edgelist", "graph_node_id2idx.txt",
                 "graph_A.csv", "graph_A.zip", "graph_A.npy", "road_embed.txt"):
        if (graph / name).exists():
            (graph / name).unlink()
            print(f"  removed stale graph/{name}")


def step_graph() -> None:
    in_model_dir("python", "build_graph_nyc.py", "--data-dir", "./data/NYC")


def step_embed() -> None:
    if EMBED_CPU:
        in_model_dir("python", "train_road_embed.py", "--data-dir", "./data/NYC", "--device", "cpu")
    else:
        in_model_dir("python", "train_road_embed.py", "--data-dir", "./data/NYC", gpu=True)


RUNNERS = {"prep": step_prep, "ubodt": step_ubodt, "fmm": step_fmm, "convert": step_convert,
           "graph": step_graph, "embed": step_embed}


# Main
def main() -> None:
    global DRY_RUN
    p = argparse.ArgumentParser(description="Run the NYC Diff-RNTraj FMM pipeline.")
    p.add_argument("--from", dest="start", choices=STEPS, default="prep", help="first step (default: prep)")
    p.add_argument("--to", dest="end", choices=STEPS, default=DEFAULT_LAST, help=f"last step (default: {DEFAULT_LAST})")
    p.add_argument("--only", choices=STEPS, help="run just this one step")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("--embed-cpu", action="store_true",
                   help="run the road embedding on the CPU (no NVIDIA GPU; much slower)")
    a = p.parse_args()
    DRY_RUN = a.dry_run
    global EMBED_CPU
    EMBED_CPU = a.embed_cpu
    if not DRY_RUN:
        start_pipeline_log()

    first, last = (a.only, a.only) if a.only else (a.start, a.end)
    steps = STEPS[STEPS.index(first): STEPS.index(last) + 1]
    if not steps:
        sys.exit(f"--from {first} comes after --to {last}")

    banner(f"NYC Diff-RNTraj pipeline: {' -> '.join(steps)}")
    print(f"pipeline folder    : {HERE}")
    print(f"road network (FMM) : {FMM_NETWORK}")
    print(f"GPS points         : {GPS_CSV}")
    print(f"UBODT              : {UBODT_BIN}")
    print(f"FMM matches        : {MATCH_CSV}")
    print(f"model data folder  : {MODEL_DATA}")
    if not DRY_RUN:
        print(f"log                : {PIPELINE_LOG}")
    print(f"image              : {FMM_IMAGE} ({FMM_COMPOSE})")
    if "embed" in steps:
        print(f"road embedding on  : {'CPU (--embed-cpu)' if EMBED_CPU else 'GPU'}")

    # checks before spending time on anything
    if not DRY_RUN:
        if shutil.which("docker") is None:
            sys.exit("Docker not found. Start Docker Desktop and try again.")
        if "prep" in steps and not (HERE / PARQUET).exists():
            sys.exit(f"Parquet not found: {(HERE / PARQUET).resolve()}")
        if not docker_image_exists(FMM_IMAGE):
            heading(f"Building the image {FMM_IMAGE} (first time only: compiles FMM, takes a while)")
            run(["docker", "compose", "-f", str(FMM_COMPOSE), "build"])

    t_all = time.time()
    for i, name in enumerate(steps, 1):
        heading(f"[{i}/{len(steps)}] {name}")
        t = time.time()
        try:
            RUNNERS[name]()
        except Exception as exc:
            print(f"\n❌ Step '{name}' failed: {exc}")
            print(f"   Fix the problem, then resume with:  python {Path(__file__).name} --from {name}")
            sys.exit(1)
        print(f"✅ {name} done in {(time.time() - t) / 60:.1f} min")

    heading(f"All done in {(time.time() - t_all) / 60:.1f} min")
    if "embed" not in steps and "graph" in steps:
        print("Next: road_embed.txt (slow) ->  python run_diffrntraj_pipeline.py --only embed")


if __name__ == "__main__":
    main()
