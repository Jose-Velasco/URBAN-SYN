"""Compose named Hydra presets for local and Slurm experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# HPC scripts and tests load this file directly rather than importing an hpc package.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config_validation import (
    GNNArchitecture, ConnectivityObjective, GNNStrategy,
    PresetTags, RoadEncoderChoice, LossChoice, SmokeBudget,
    validate_composed_experiment, validate_preset_groups, validate_preset_name
)


def compose_experiment(config_dir: Path, preset: str) -> dict[str, Any]:
    """Compose a named preset using Hydra and validate its resolved settings.

    Args:
        config_dir: Directory containing base.yaml and configuration groups.
        preset: Explicit experiment name chosen for submission.

    Returns:
        Plain dictionary suitable for the existing Pydantic experiment loader.
    """
    config_dir = Path(config_dir).expanduser().resolve()
    validate_preset_name(config_dir, preset)

    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError("Hydra composition requires hydra-core. Install it before using presets.") from error

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        composed = compose(config_name="base", overrides=[f"preset={preset}"])
    values = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(values, dict):
        raise ValueError("The composed experiment must be a YAML mapping.")
    validate_composed_experiment(values, preset, config_dir.parents[2])
    validate_preset_groups(config_dir, preset, values)
    return values


def main() -> None:
    """Preview a composed configuration without modifying files or scheduling jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--preset", required=True)
    args = parser.parse_args()
    try:
        resolved = compose_experiment(args.config_dir, args.preset)
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(yaml.safe_dump(resolved, sort_keys=False))


if __name__ == "__main__":
    main()
