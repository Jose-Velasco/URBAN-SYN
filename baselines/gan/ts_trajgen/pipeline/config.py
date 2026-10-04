from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_GENERATION_MODES = {"pretrained", "gan", "both"}


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration used by the TS-TrajGen wrapper.

    This configuration is intentionally separate from the model/training YAML
    consumed by the adapted TS-TrajGen source code. The wrapper configuration
    only describes orchestration concerns such as dataset paths, runtime
    settings, preprocessing options, graph partitioning, and generation mode.
    """

    # Dataset
    dataset_name: str

    # Wrapper/container paths
    workspace_dir: Path
    repo_dir: Path
    datasets_dir: Path
    outputs_dir: Path

    # Upstream read-only inputs
    network_path: Path
    fmm_match_path: Path
    parquet_path: Path
    trip_id_map_path: Path

    # TS-TrajGen experiment/model configuration
    model_config_path: Path

    # Runtime
    device: str

    # build_tstrajgen_inputs.py
    min_len: int
    min_delta_seconds: float
    train_ratio: float
    interpolate_intermediate_edges: bool
    max_train_trajectories: int | None
    max_test_trajectories: int | None

    # KaHIP
    partition_k: int
    partition_preconfiguration: str

    # GAN execution options
    gan_exp_id: int
    gan_debug: bool
    pretrain_discriminator: bool

    # Generation
    generation_mode: str

    @property
    def dataset_dir(self) -> Path:
        """Directory containing artifacts for the selected dataset."""
        return self.datasets_dir / self.dataset_name

    @property
    def log_dir(self) -> Path:
        """Directory used by wrapper-owned preprocessing logs."""
        return self.datasets_dir / "logs"

    @property
    def road_save_dir(self) -> Path:
        """Directory containing road/region pretraining checkpoints."""
        return self.repo_dir / "save" / self.dataset_name

    @property
    def road_gan_save_dir(self) -> Path:
        """Directory containing road-level GAN checkpoints."""
        return self.road_save_dir / "gan"

    @property
    def region_gan_save_dir(self) -> Path:
        """Directory containing region-level GAN checkpoints."""
        return self.road_save_dir / "region_gan"

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PipelineConfig":
        """Load and validate a wrapper configuration from YAML.

        Relative paths are resolved relative to the wrapper YAML file itself.
        Absolute container paths such as ``/workspace/repo`` are preserved.

        Args:
            config_path:
                Path to the wrapper YAML configuration.

        Returns:
            Parsed and validated ``PipelineConfig``.

        Raises:
            FileNotFoundError:
                If the YAML file does not exist.
            ValueError:
                If required configuration values are missing or invalid.
            TypeError:
                If the top-level YAML document is not a mapping.
        """
        config_path = config_path.expanduser().resolve()

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Pipeline configuration does not exist: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise TypeError(
                "Pipeline configuration must contain a YAML mapping at the top level."
            )

        base_dir = config_path.parent

        dataset = _require_mapping(raw, "dataset")
        paths = _require_mapping(raw, "paths")
        runtime = _require_mapping(raw, "runtime")
        build = _require_mapping(raw, "build")
        partition = _require_mapping(raw, "partition")
        gan = _require_mapping(raw, "gan")
        generation = _require_mapping(raw, "generation")

        config = cls(
            dataset_name=_require_str(dataset, "name"),
            workspace_dir=_resolve_path(
                _require_value(paths, "workspace"), base_dir
            ),
            repo_dir=_resolve_path(
                _require_value(paths, "repo"), base_dir
            ),
            datasets_dir=_resolve_path(
                _require_value(paths, "datasets"), base_dir
            ),
            outputs_dir=_resolve_path(
                _require_value(paths, "outputs"), base_dir
            ),
            network_path=_resolve_path(
                _require_value(paths, "network"), base_dir
            ),
            fmm_match_path=_resolve_path(
                _require_value(paths, "fmm_match"), base_dir
            ),
            parquet_path=_resolve_path(
                _require_value(paths, "parquet"), base_dir
            ),
            trip_id_map_path=_resolve_path(
                _require_value(paths, "trip_id_map"), base_dir
            ),
            model_config_path=_resolve_path(
                _require_value(raw, "model_config"), base_dir
            ),
            device=_require_str(runtime, "device"),
            min_len=_require_int(build, "min_len"),
            min_delta_seconds=_require_float(build, "min_delta_seconds"),
            train_ratio=_require_float(build, "train_ratio"),
            interpolate_intermediate_edges=_require_bool(
                build, "interpolate_intermediate_edges"
            ),
            max_train_trajectories=_optional_positive_int(
                build, "max_train_trajectories"
            ),
            max_test_trajectories=_optional_positive_int(
                build, "max_test_trajectories"
            ),
            partition_k=_require_int(partition, "k"),
            partition_preconfiguration=_require_str(
                partition, "preconfiguration"
            ),
            gan_exp_id=_require_int(gan, "exp_id"),
            gan_debug=_require_bool(gan, "debug"),
            pretrain_discriminator=_require_bool(
                gan, "pretrain_discriminator"
            ),
            generation_mode=_require_str(generation, "mode").lower(),
        )

        config.validate()
        return config

    def validate(self) -> None:
        """Validate semantic constraints that are not covered by YAML parsing."""
        if not self.dataset_name.strip():
            raise ValueError("dataset.name cannot be empty.")

        if self.min_len < 1:
            raise ValueError("build.min_len must be at least 1.")

        if self.min_delta_seconds < 0:
            raise ValueError("build.min_delta_seconds cannot be negative.")

        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("build.train_ratio must be between 0 and 1.")

        if self.partition_k < 2:
            raise ValueError("partition.k must be at least 2.")

        if self.gan_exp_id < 0:
            raise ValueError("gan.exp_id cannot be negative.")

        if self.generation_mode not in _GENERATION_MODES:
            allowed = ", ".join(sorted(_GENERATION_MODES))
            raise ValueError(
                f"generation.mode must be one of: {allowed}. "
                f"Got: {self.generation_mode!r}"
            )


def load_pipeline_config(config_path: str | Path) -> PipelineConfig:
    """Convenience wrapper for loading a pipeline YAML file."""
    return PipelineConfig.from_yaml(Path(config_path))


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _require_value(mapping, key)

    if not isinstance(value, dict):
        raise TypeError(f"{key!r} must be a YAML mapping.")

    return value


def _require_value(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required pipeline configuration key: {key}")

    return mapping[key]


def _require_str(mapping: dict[str, Any], key: str) -> str:
    value = _require_value(mapping, key)

    if not isinstance(value, str):
        raise TypeError(f"{key!r} must be a string.")

    value = value.strip()

    if not value:
        raise ValueError(f"{key!r} cannot be empty.")

    return value


def _require_int(mapping: dict[str, Any], key: str) -> int:
    value = _require_value(mapping, key)

    # bool is a subclass of int, so explicitly reject it.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key!r} must be an integer.")

    return value

def _optional_positive_int(
    mapping: dict[str, Any],
    key: str,
) -> int | None:
    """Read an optional positive integer from a YAML mapping."""
    value = mapping.get(key)

    if value is None:
        return None

    # bool is a subclass of int, so reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key!r} must be an integer or null.")

    if value <= 0:
        raise ValueError(f"{key!r} must be greater than 0.")

    return value


def _require_float(mapping: dict[str, Any], key: str) -> float:
    value = _require_value(mapping, key)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key!r} must be a number.")

    return float(value)


def _require_bool(mapping: dict[str, Any], key: str) -> bool:
    value = _require_value(mapping, key)

    if not isinstance(value, bool):
        raise TypeError(f"{key!r} must be true or false.")

    return value


def _resolve_path(value: Any, base_dir: Path) -> Path:
    """Convert a YAML path value to an absolute ``Path``."""
    if not isinstance(value, (str, Path)):
        raise TypeError(f"Expected a filesystem path, got {type(value).__name__}.")

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = base_dir / path

    return path.resolve()
