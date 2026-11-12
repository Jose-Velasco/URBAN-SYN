"""Read eval_config.yaml and check that every input file is where the config says it is."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("eval_config.yaml")


class MissingInputsError(Exception):
    """Raised when one or more input files are missing; the message lists all of them."""


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load the YAML config and turn repo_root into an absolute path.

    repo_root in the file is relative to the config file itself, so the same config works wherever the repo is cloned.
    """
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config.get("repo_root", ".")).expanduser()
    config["repo_root"] = root if root.is_absolute() else (path.parent / root).resolve()
    config["config_path"] = path
    return config


def model_names(config: dict) -> list[str]:
    """Return the names of all models listed in the config, in file order."""
    return list(config["models"])


def model_files(config: dict, model: str) -> dict[str, Path]:
    """Return {file key: absolute path} for one model's input files."""
    if model not in config["models"]:
        raise KeyError(f"Unknown model '{model}'. Known models: {', '.join(model_names(config))}")
    root = config["repo_root"]
    files = config["models"][model].get("files", {})
    return {key: _resolve(root, value) for key, value in files.items()}


def model_options(config: dict, model: str) -> dict:
    """Return the optional converter settings for one model (empty dict if none)."""
    return config["models"][model].get("options", {}) or {}


def network_files(config: dict, network: str) -> dict[str, Path]:
    """Return {file key: absolute path} for one road network listed under `networks:`."""
    if network not in config.get("networks", {}):
        raise KeyError(f"Unknown road network '{network}'. Add it under networks: in eval_config.yaml")
    root = config["repo_root"]
    return {key: _resolve(root, value) for key, value in config["networks"][network]["files"].items()}


def resolve(config: dict, value: str) -> Path:
    """An absolute path for a path written in the config (relative paths start at repo_root)."""
    return _resolve(config["repo_root"], value)


def fmm_settings(config: dict) -> dict:
    """The fmm: section with absolute paths, plus the road network and shortest-path table its template uses.

    The template is TS-TrajGen's FMM config; its relative paths (network, UBODT) start at workdir, where FMM runs.
    """
    section = config.get("fmm")
    if not section:
        raise SystemExit("eval_config.yaml has no fmm: section, which the models with an fmm_dir need")
    template = resolve(config, section["template"])
    workdir = resolve(config, section.get("workdir", "."))
    settings = {"template": template, "workdir": workdir, "command": section.get("command", "fmm")}
    if template.exists():
        text = template.read_text(encoding="utf-8")
        root = ET.fromstring(text)
        settings["template_text"] = text
        settings["network"] = (workdir / root.findtext("input/network/file").strip()).resolve()
        settings["ubodt"] = (workdir / root.findtext("input/ubodt/file").strip()).resolve()
    return settings


def fmm_files(config: dict) -> list[tuple[str, Path, str]]:
    """(key, path, produced by) of the files FMM matching needs: its config template, road network and UBODT."""
    settings = fmm_settings(config)
    produced_by = config["fmm"].get("produced_by", {})
    files = [("template", settings["template"])]
    if "network" in settings:
        network = settings["network"]
        files += [("network", network)] + [(f"network {s}", network.with_suffix(s)) for s in (".dbf", ".shx")]
        files += [("ubodt", settings["ubodt"])]
    return [(key, path, produced_by.get(key.split()[0], "unknown")) for key, path in files]


def check_inputs(config: dict, models: list[str], networks: list[str] = (), fmm: bool = True) -> None:
    """Check every input file of the given models (and road networks) and raise MissingInputsError if any is missing.

    All missing files are collected first, so one run reports every problem instead of stopping at the first one. With
    fmm=False (--no-fmm), FMM's own files are not needed.
    """
    problems = []
    for model in models:
        produced_by = config["models"][model].get("produced_by", {})
        for key, path in model_files(config, model).items():
            if not path.exists():
                problems.append((model, key, path, produced_by.get(key, "unknown")))
    if fmm and any("fmm_dir" in config["models"][m] for m in models):
        problems += [("FMM matching", key, path, by) for key, path, by in fmm_files(config) if not path.exists()]
    for network in networks:
        produced_by = config["networks"][network].get("produced_by", {})
        for key, path in network_files(config, network).items():
            if not path.exists():
                problems.append((f"network {network}", key, path, produced_by.get(key, "unknown")))
    if problems:
        raise MissingInputsError(format_missing(problems, config["repo_root"]))


def format_missing(problems: list[tuple], repo_root: Path) -> str:
    """Build the missing-file error text: one block per file, then what to do next."""
    width = max(13, max(len(name) for name, *_ in problems) + 2)
    lines = [f"{len(problems)} input file(s) missing; nothing was converted.", ""]
    for name, key, path, produced_by in problems:
        shown = path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
        lines.append(f"  {name:<{width}}{shown.as_posix()}")
        lines.append(f"  {'':<{width}}file: {key}")
        lines.append(f"  {'':<{width}}produced by: {produced_by}")
        lines.append("")
    models = sorted({name for name, *_ in problems if not name.startswith(("network ", "FMM "))})
    if models:
        lines.append("Fix the paths in eval_config.yaml, or leave out these models: " + " ".join(models))
    else:
        lines.append("Fix the paths under networks: or fmm: in eval_config.yaml")
    return "\n".join(lines)


def _resolve(root: Path, value: str) -> Path:
    """Turn a config path into an absolute path (relative paths start at repo_root)."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path
