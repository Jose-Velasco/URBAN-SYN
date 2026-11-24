"""Pre-submission validation for Hydra presets and study-based experiments.

Only called from development/submission Python. Compute-node checkpoint handoff
and snapshot verification remain standard-library-only in their own scripts.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator, model_validator


# Study catalog and Slurm settings

SafeName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
SingleLine = Annotated[str, StringConstraints(pattern=r"^[^\r\n]*$")]
HPCValue = SingleLine | int | None


class HPCSettings(BaseModel):
    """Slurm defaults declared by the study before environment overrides."""

    model_config = ConfigDict(extra="forbid", strict=True)

    partition: HPCValue = None
    gres: HPCValue = None
    time: HPCValue = None
    memory: HPCValue = None
    cpus_per_task: HPCValue = None
    mail_user: HPCValue = None
    mail_type: HPCValue = None
    exclude_nodes: list[str] = Field(default_factory=list)

    @field_validator("exclude_nodes", mode="before")
    @classmethod
    def validate_excluded_nodes(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("hpc.exclude_nodes must be a list of node names.")
        if any(not isinstance(node, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", node) for node in value):
            raise ValueError("hpc.exclude_nodes contains an invalid node name.")
        return value

    def shell_value(self, field: str) -> str:
        value = getattr(self, field)
        return "" if value is None else str(value)


class StudyExperiment(BaseModel):
    """A direct experiment config or a named Hydra preset."""

    model_config = ConfigDict(extra="forbid", strict=True)

    config: str | None = None
    preset: SafeName | None = None
    pretrained_from: SafeName | None = None
    source_study: SafeName | None = None

    @model_validator(mode="after")
    def one_configuration_source(self) -> StudyExperiment:
        if (self.config is None) == (self.preset is None):
            raise ValueError("Study entries must select exactly one of 'config' or 'preset'.")
        return self


class StudySettings(BaseModel):
    """Study identity, experiment catalog, and shared graph setting."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: SafeName
    experiments: dict[SafeName, str | StudyExperiment]
    graph_config: str | None = None
    composition: str | None = None


class StudyFile(BaseModel):
    """The complete study document and its allowed top-level keys."""

    model_config = ConfigDict(extra="forbid", strict=True)

    study: StudySettings
    hpc: HPCSettings = Field(default_factory=HPCSettings)


class PublishedEncoderReference(BaseModel):
    """Identity and checkpoint location recorded by a completed pretraining job."""

    model_config = ConfigDict(extra="allow", strict=True)

    format_version: Literal[1]
    study_name: str
    experiment_key: str
    run_id: str = Field(min_length=1)
    run_dir: str
    encoder_checkpoint: str


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a mapping from a YAML file, with a helpful error on bad input."""
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def inside_project(value: str | Path, root: Path) -> Path:
    """Resolve an experiment path while preventing access outside the checkout."""
    supplied = Path(value).expanduser()
    path = (supplied if supplied.is_absolute() else root / supplied).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Experiment files must be inside the project: {path}")
    return path


def detect_pipeline(experiment: dict[str, Any], config_path: Path) -> str:
    """Distinguish link pretraining from joint diffusion by configuration shape."""
    pretrain = "pretraining" in experiment
    joint = isinstance(experiment.get("model"), dict) and "diffusion" in experiment["model"]
    if pretrain == joint:
        raise ValueError(f"Cannot distinguish link pretraining from joint diffusion: {config_path}")
    return "link_pretrain" if pretrain else "joint"


def validate_study_entry(entry: StudyExperiment, name: str) -> None:
    if entry.preset is not None and entry.preset != name:
        raise ValueError("A Hydra preset must match its study experiment key and cannot select a second YAML.")


def validate_study_graph(root: Path, study_graph: str | None, experiment: dict[str, Any], name: str) -> None:
    """Keep the shared graph-file constraint for older study entries."""
    if study_graph is None:
        return
    selected = (experiment.get("paths") or {}).get("graph_config")
    if not isinstance(selected, str) or inside_project(study_graph, root) != inside_project(selected, root):
        raise ValueError(f"Graph YAML differs from the study for {name}: {selected}")


# Hydra choices and cross-field consistency

class GNNArchitecture(StrEnum):
    GINE = "gine"
    GATV2 = "gatv2"


class ConnectivityObjective(StrEnum):
    NONE = "none"
    BCE = "bce"
    BPR = "bpr"


class GNNStrategy(StrEnum):
    JOINT = "joint"
    FROZEN = "frozen"
    WARMUP = "warmup"


class PresetTags(BaseModel):
    """Ablation choices recorded in the composed experiment's tracking tags."""

    model_config = ConfigDict(extra="ignore")

    gnn_architecture: GNNArchitecture
    pretraining_objective: ConnectivityObjective
    connectivity_loss: ConnectivityObjective
    gnn_update: GNNStrategy
    profile: str | None = None

    @property
    def experiment_name(self) -> str:
        if self.connectivity_loss == ConnectivityObjective.NONE:
            variant = "baseline_warmup" if self.gnn_update == GNNStrategy.WARMUP else "baseline"
        else:
            variant = f"{self.connectivity_loss}_{self.gnn_update.value}"
        suffix = "_smoke" if self.profile == "smoke" else ""
        return f"{self.gnn_architecture}_{variant}{suffix}"


class WarmupChoice(BaseModel):
    """Read the warmup toggle; the joint experiment owns schedule parameters."""

    model_config = ConfigDict(extra="allow", strict=True)

    enabled: bool = False


class RoadEncoderChoice(BaseModel):
    """Encoder settings relevant to preset consistency checks."""

    model_config = ConfigDict(extra="ignore", strict=True)

    initialization: Literal["random", "pretrained"]
    trainable: bool
    checkpoint: str | None = None
    gradient_warmup: WarmupChoice | None = None

    @property
    def warmup_enabled(self) -> bool:
        return self.gradient_warmup.enabled if self.gradient_warmup is not None else False


class LossChoice(BaseModel):
    """Connectivity objective flags and their optimizer contribution."""

    model_config = ConfigDict(extra="ignore", strict=True)

    enabled: bool = False
    include_in_total: bool = False
    weight: float = 0.0


class SmokeBudget(BaseModel):
    """Bound small smoke runs by the total number of training minibatches."""

    model_config = ConfigDict(extra="ignore", strict=True)

    epochs: int = Field(gt=0)
    max_batches: int = Field(gt=0)

    @model_validator(mode="after")
    def limit_total_work(self) -> SmokeBudget:
        if self.epochs * self.max_batches > 100:
            raise ValueError("Smoke run exceeds 100 training batches.")
        return self


def validate_preset_name(config_dir: Path, preset: str) -> None:
    """Require an explicitly named preset whose source file exists."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", preset):
        raise ValueError("Preset must be a lowercase name containing letters, numbers, or underscores.")
    if not (config_dir / "preset" / f"{preset}.yaml").is_file():
        raise ValueError(f"Unknown experiment preset '{preset}' in {config_dir / 'preset'}.")


def _validate_preset_graph(config: dict[str, Any], tags: PresetTags, root: Path, preset: str) -> None:
    graph = read_yaml_mapping(inside_project(config["paths"]["graph_config"], root))
    if graph["model"]["gnn"]["architecture"] != tags.gnn_architecture:
        raise ValueError(f"Preset '{preset}' selects {tags.gnn_architecture} but graph YAML uses a different GNN.")
    if tags.gnn_architecture == GNNArchitecture.GINE and not graph["model"]["edge_features"]:
        raise ValueError("GINE requires selected edge features in the road graph configuration.")


def _validate_preset_encoder(config: dict[str, Any], tags: PresetTags) -> None:
    encoder = RoadEncoderChoice.model_validate(config["model"]["road_encoder"])
    scorer = config["model"].get("connectivity_scorer", {})
    is_baseline = tags.connectivity_loss == ConnectivityObjective.NONE
    warmup = tags.gnn_update == GNNStrategy.WARMUP
    if is_baseline:
        if (tags.pretraining_objective != ConnectivityObjective.NONE or
                tags.gnn_update not in (GNNStrategy.JOINT, GNNStrategy.WARMUP) or
                encoder.initialization != "random" or not encoder.trainable or
                encoder.warmup_enabled != warmup):
            raise ValueError("A baseline requires a trainable random GNN, with optional gradient warmup.")
        if scorer.get("enabled", False) or config["tracking"]["logging"]["scorer"]["enabled"]:
            raise ValueError("A baseline must not enable a connectivity scorer.")
    else:
        if tags.pretraining_objective != tags.connectivity_loss:
            raise ValueError("Pretraining objective must match the connectivity loss.")
        if tags.gnn_update not in (GNNStrategy.FROZEN, GNNStrategy.WARMUP):
            raise ValueError("Connectivity-guided presets require a pretrained frozen or warmup GNN.")
        if encoder.initialization != "pretrained":
            raise ValueError("GNN initialization does not match the selected strategy.")
        if encoder.trainable != warmup or encoder.warmup_enabled != warmup:
            raise ValueError("GNN trainability and gradient warmup must match the selected strategy.")
        if scorer != {"enabled": True, "checkpoint": "auto", "trainable": False}:
            raise ValueError("Connectivity regularization requires the matching frozen pretrained scorer.")
    if encoder.checkpoint is not None:
        raise ValueError("The encoder checkpoint must be supplied by the submission resolver.")


def _validate_preset_losses(config: dict[str, Any], tags: PresetTags) -> None:
    for objective in (ConnectivityObjective.BCE, ConnectivityObjective.BPR):
        loss = LossChoice.model_validate(config["training"]["losses"][f"connectivity_{objective}"])
        if loss.enabled != (tags.connectivity_loss == objective):
            raise ValueError("Enabled connectivity loss does not match the selected objective.")
        if loss.enabled and (not loss.include_in_total or loss.weight <= 0):
            raise ValueError("Selected connectivity loss must contribute a positive training weight.")


def validate_composed_experiment(config: dict[str, Any], preset: str, project_root: Path) -> None:
    """Validate a composed preset's metadata, graph, model, losses, and smoke budget.

    Args:
        config: Fully composed Hydra experiment dictionary.
        preset: Chosen experiment preset name.
        project_root: Root used to resolve referenced graph configurations.
    """
    tags = PresetTags.model_validate(config["tracking"]["tags"])
    if tags.profile not in (None, "smoke"):
        raise ValueError("Unknown experiment profile.")
    if preset != tags.experiment_name:
        raise ValueError(f"Preset '{preset}' does not match its resolved settings ('{tags.experiment_name}').")
    _validate_preset_graph(config, tags, project_root, preset)
    _validate_preset_encoder(config, tags)
    _validate_preset_losses(config, tags)
    if tags.profile == "smoke":
        try:
            SmokeBudget.model_validate(config["training"])
        except ValidationError as error:
            raise ValueError(
                "Smoke presets require a finite budget of at most 100 training batches "
                "(training.epochs * training.max_batches)."
            ) from error


def validate_preset_groups(config_dir: Path, preset: str, config: dict[str, Any]) -> None:
    """Ensure Hydra's actual selected groups agree with the resolved metadata."""
    source = read_yaml_mapping(config_dir / "preset" / f"{preset}.yaml")
    selected = {
        key.removeprefix("/"): value
        for entry in source["defaults"] if isinstance(entry, dict)
        for key, value in entry.items()
    }
    tags = PresetTags.model_validate(config["tracking"]["tags"])
    required = {
        "gnn": tags.gnn_architecture.value,
        "encoder": (
            "warmup_random" if tags.connectivity_loss == ConnectivityObjective.NONE
            and tags.gnn_update == GNNStrategy.WARMUP else
            "scratch" if tags.gnn_update == GNNStrategy.JOINT else tags.gnn_update.value
        ),
        "pretraining": tags.pretraining_objective.value,
        "connectivity": tags.connectivity_loss.value
    }
    if tags.profile == "smoke":
        required["profile"] = "smoke"
    if selected != required:
        raise ValueError(f"Preset '{preset}' selects groups inconsistent with its resolved settings.")


# Pretraining dependencies and local workflows

def validate_source_graph(
    root: Path, settings: StudySettings, source: dict[str, Any], joint: dict[str, Any],
    source_name: str, hydra_preset: bool
) -> None:
    """Check graph identity for legacy entries or graph/feature equivalence for Hydra."""
    source_graph = (source.get("paths") or {}).get("graph_config")
    if hydra_preset:
        source_yaml = read_yaml_mapping(inside_project(source_graph, root))
        joint_yaml = read_yaml_mapping(inside_project(joint["paths"]["graph_config"], root))
        if source_yaml != joint_yaml:
            raise ValueError(f"Pretraining source '{source_name}' has incompatible graph/GNN settings.")
        objective = (joint.get("tracking") or {}).get("tags", {}).get("pretraining_objective")
        if objective is not None and source["pretraining"]["objective"] != objective:
            raise ValueError(f"Pretraining source '{source_name}' has a mismatched BCE/BPR objective.")
        return
    if (not isinstance(source_graph, str) or settings.graph_config is None or
            inside_project(source_graph, root) != inside_project(settings.graph_config, root)):
        raise ValueError(f"Pretraining source '{source_name}' must use the study's graph YAML.")


def validate_encoder_source(joint: dict[str, Any]) -> None:
    """Ensure a study-provided encoder cannot conflict with a YAML checkpoint path."""
    encoder = (joint.get("model") or {}).get("road_encoder") or {}
    initialization = encoder.get("initialization")
    scorer = (joint.get("model") or {}).get("connectivity_scorer") or {}
    if initialization != "pretrained" and not (
        initialization == "random" and encoder.get("trainable") is True
        and scorer.get("enabled") is True and scorer.get("checkpoint") == "auto"
        and scorer.get("trainable") is False
    ):
        raise ValueError("Study pretrained_from requires model.road_encoder.initialization=pretrained "
                         "or a scratch GNN with a frozen pretrained scorer.")
    if encoder.get("checkpoint") is not None:
        raise ValueError(
            "Set model.road_encoder.checkpoint: null when using study pretrained_from "
            "to avoid two competing checkpoint selections."
        )


def validate_pretraining_dependency(pipeline: str, settings: StudySettings | None, study: str | None) -> None:
    if pipeline != "joint":
        raise ValueError("pretrained_from is only supported for joint diffusion experiments.")
    if settings is None or study is None:
        raise ValueError("Pretraining dependencies require a study catalog.")


def validate_published_encoder_reference(
    root: Path, reference: dict[str, Any], reference_path: Path, source_study: str, source_experiment: str
) -> tuple[str, str]:
    """Validate a published encoder's identity and exact run-owned checkpoint path."""
    try:
        saved = PublishedEncoderReference.model_validate(reference)
    except ValidationError as error:
        raise ValueError(f"Invalid or incomplete pretraining reference: {reference_path}") from error
    if saved.study_name != source_study or saved.experiment_key != source_experiment:
        raise ValueError(f"Invalid or mismatched pretraining reference: {reference_path}")
    run_path = inside_project(saved.run_dir, root)
    selected_path = inside_project(saved.encoder_checkpoint, root)
    if selected_path != run_path / "checkpoints/encoder.pt":
        raise ValueError(f"Pretraining reference has an unexpected encoder path: {reference_path}")
    if run_path.name != saved.run_id or not selected_path.is_file():
        raise FileNotFoundError(f"Selected pretrained encoder is missing or does not match its run: {selected_path}")
    return str(selected_path), saved.run_id


def validate_local_request(selected: Any, stage: str, encoder_checkpoint: str | None) -> None:
    """Check stage and checkpoint choices before creating any local run files."""
    if selected.pipeline == "joint" and selected.composed_config is None:
        raise ValueError("Local joint runs require a named Hydra preset, not a legacy experiment YAML.")
    if selected.pipeline == "link_pretrain" and stage == "joint":
        raise ValueError("A pretraining experiment does not have a joint diffusion stage.")
    if selected.pipeline == "joint" and stage == "pretrain" and not selected.pretrain_config:
        raise ValueError("This baseline does not require pretraining.")
    if encoder_checkpoint and (not selected.pretrained_from or stage == "pretrain"):
        raise ValueError("--encoder-checkpoint is only for a pretrained joint experiment.")
    if selected.pretrained_from and stage == "joint" and not encoder_checkpoint:
        raise ValueError("Joint-only execution requires --encoder-checkpoint from a completed pretraining run.")


def validate_local_encoder(root: Path, checkpoint: str | Path) -> Path:
    """Require the encoder and its paired frozen scorer to come from one run."""
    path = inside_project(checkpoint, root)
    if path.name != "encoder.pt" or path.parent.name != "checkpoints":
        raise ValueError("Use a pretraining run's checkpoints/encoder.pt file.")
    if not path.is_file() or not path.with_name("best.pt").is_file():
        raise FileNotFoundError(f"Expected encoder.pt and best.pt from the same pretraining run: {path.parent}")
    return path
