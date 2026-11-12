"""Convert baselines into the common format and check the result.

Run from the URBAN-SYN folder:
    python -m evaluation.convert --model tstrajgen
    python -m evaluation.convert --model all --check        # only check that the input files exist
    python -m evaluation.convert --model difftraj --save evaluation/results/converted
    python -m evaluation.convert --model lstm_trajgan --no-fmm          # skip FMM map matching

DiffTraj and LSTM-TrajGAN get road tables from FMM, run here when its matches are missing or stale (fmm_prepare.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from evaluation import fmm_prepare, inputs, tables
from evaluation.converters import CONVERTERS


def parse_args(argv=None) -> argparse.Namespace:
    """Read the command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert baseline outputs into the common format.")
    parser.add_argument("--model", nargs="+", required=True,
                        help=f"model name(s) to convert, or 'all': {', '.join(CONVERTERS)}")
    parser.add_argument("--config", default=inputs.DEFAULT_CONFIG, help="path to eval_config.yaml")
    parser.add_argument("--check", action="store_true", help="only check that the input files exist")
    parser.add_argument("--save", type=Path, default=None,
                        help="also write the tables as CSV files to this folder, one subfolder per model")
    parser.add_argument("--no-fmm", action="store_true",
                        help="don't map-match the GPS models (DiffTraj, LSTM-TrajGAN): no road tables for them")
    return parser.parse_args(argv)


def chosen_models(requested: list[str], config: dict) -> list[str]:
    """Expand 'all' and check that every requested model has a converter and a config entry."""
    models = list(CONVERTERS) if requested == ["all"] else requested
    unknown = [m for m in models if m not in CONVERTERS or m not in config["models"]]
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(unknown)}. Choose from: {', '.join(CONVERTERS)}, all")
    return models


def model_network(model: str) -> str:
    """The road network a model's road tables use (the NETWORK constant of its converter module)."""
    return sys.modules[CONVERTERS[model].__module__].NETWORK


def convert_model(config: dict, model: str, fmm: bool = True, module: str = "convert", args: str = "",
                  log=print) -> tables.Converted:
    """Run one model's converter on the files listed for it in the config.

    For the GPS models (those with an fmm_dir), road tables are then made from FMM matches unless fmm is False.
    """
    files = inputs.model_files(config, model)
    options = inputs.model_options(config, model)
    result = CONVERTERS[model](files, options)
    if fmm and fmm_prepare.uses_fmm(config, model):
        fmm_prepare.add_road_tables(config, model, result, model_network(model), module, args, log)
    return result


def print_summary(result: tables.Converted, problems: list[str], seconds: float) -> None:
    """Print what a converter produced, its notes and any problems the checks found."""
    print(f"\n{result.model}  (one trajectory = one {result.unit}, {seconds:.1f} s)")
    print(f"  generated        {tables.summarize(result.generated)}")
    print(f"  real test        {tables.summarize(result.real)}")
    print(f"  real train       {tables.summarize(result.train)}")
    print(f"  generated roads  {tables.summarize(result.generated_roads)}")
    print(f"  real roads       {tables.summarize(result.real_roads)}")
    refs = result.generated.drop_duplicates("traj_id")["ref_id"].notna().sum()
    print(f"  paired by ID     {refs:,} generated trajectories have a ref_id")
    for key, value in result.notes.items():
        print(f"  note: {key}: {value}")
    print("  checks: " + ("all passed" if not problems else "PROBLEMS"))
    for problem in problems:
        print(f"    - {problem}")


def save_tables(result: tables.Converted, folder: Path) -> None:
    """Write each available table as <folder>/<model>/<table>.csv."""
    out = folder / result.model
    out.mkdir(parents=True, exist_ok=True)
    for name in ("generated", "real", "train", "generated_roads", "real_roads"):
        table = getattr(result, name)
        if table is not None:
            table.to_csv(out / f"{name}.csv", index=False)
    print(f"  saved to {out}")


def main(argv=None) -> int:
    """Check the inputs, then convert, check and summarize each requested model."""
    args = parse_args(argv)
    config = inputs.load_config(args.config)
    models = chosen_models(args.model, config)
    try:
        inputs.check_inputs(config, models, fmm=not args.no_fmm)
    except inputs.MissingInputsError as error:
        print(f"ERROR  {error}", file=sys.stderr)
        return 1
    print(f"All input files found for: {', '.join(models)}")
    if args.check:
        return 0

    failed = False
    for model in models:
        start = time.time()
        result = convert_model(config, model, fmm=not args.no_fmm, args=fmm_prepare.command_args(argv))
        problems = tables.validate(result)
        failed |= bool(problems)
        print_summary(result, problems, time.time() - start)
        if args.save:
            save_tables(result, args.save)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
