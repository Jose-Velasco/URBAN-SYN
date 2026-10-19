"""Load and validate the LSTM-TrajGAN wrapper YAML configuration."""

from __future__ import print_function

from pathlib import Path

import yaml


_CATEGORY_MODES = {"constant", "top10"}


class PipelineConfig(object):
    """Resolved settings used by the LSTM-TrajGAN pipeline wrapper."""

    def __init__(
        self,
        dataset_name,
        workspace_dir,
        repo_dir,
        data_dir,
        outputs_dir,
        source_path,
        train_source_path,
        test_source_path,
        builder_path,
        train_ratio,
        random_state,
        category_mode,
        constant_category,
        max_trajectories,
        tid_col,
        epochs,
        batch_size,
        save_params_rate,
        max_length,
        checkpoint_epoch,
    ):
        self.dataset_name = dataset_name
        self.workspace_dir = workspace_dir
        self.repo_dir = repo_dir
        self.data_dir = data_dir
        self.outputs_dir = outputs_dir
        self.source_path = source_path
        self.train_source_path = train_source_path
        self.test_source_path = test_source_path
        self.builder_path = builder_path

        self.train_ratio = train_ratio
        self.random_state = random_state
        self.category_mode = category_mode
        self.constant_category = constant_category
        self.max_trajectories = max_trajectories

        self.tid_col = tid_col

        self.epochs = epochs
        self.batch_size = batch_size
        self.save_params_rate = save_params_rate
        self.max_length = max_length
        self.checkpoint_epoch = checkpoint_epoch

        self.validate()

    @property
    def train_csv(self):
        return self.data_dir / "{}_train_latlon.csv".format(self.dataset_name)

    @property
    def test_csv(self):
        return self.data_dir / "{}_test_latlon.csv".format(self.dataset_name)

    @property
    def train_encoded_csv(self):
        return self.data_dir / "{}_dev_train_encoded_final.csv".format(
            self.dataset_name
        )

    @property
    def test_encoded_csv(self):
        return self.data_dir / "{}_dev_test_encoded_final.csv".format(
            self.dataset_name
        )

    @property
    def builder_metadata(self):
        return self.data_dir / "{}_lstm_trajgan_metadata.json".format(
            self.dataset_name
        )

    @property
    def train_npy(self):
        return self.data_dir / "final_train.npy"

    @property
    def test_npy(self):
        return self.data_dir / "final_test.npy"

    @property
    def checkpoint_dir(self):
        return self.outputs_dir / "training_params"

    @property
    def result_dir(self):
        return self.outputs_dir / "results"

    @property
    def generated_csv(self):
        return self.result_dir / "{}_syn_traj_test.csv".format(
            self.dataset_name
        )

    @property
    def log_dir(self):
        return self.outputs_dir / "logs"

    @property
    def csv2npy_script(self):
        return self.repo_dir / "data" / "csv2npy.py"

    @property
    def train_script(self):
        return self.repo_dir / "train.py"

    @property
    def predict_script(self):
        return self.repo_dir / "predict.py"

    @property
    def generator_checkpoint(self):
        return self.checkpoint_dir / "G_model_{}.h5".format(
            self.checkpoint_epoch
        )
    @property
    def uses_pre_split_inputs(self):
        """Return True when explicit train/test source files are configured."""
        return self.train_source_path is not None

    @property
    def input_mode(self):
        """Return the configured dataset input mode."""
        if self.uses_pre_split_inputs:
            return "pre_split"

        return "single_source_split"

    @property
    def source_paths(self):
        """Return all source files required by the build stage."""
        if self.uses_pre_split_inputs:
            return (
                self.train_source_path,
                self.test_source_path,
            )

        return (self.source_path,)

    def validate(self):
        """Validate the small set of settings needed by the wrapper."""
        if not self.dataset_name:
            raise ValueError("dataset.name cannot be empty.")

        self._validate_source_inputs()

        if self.category_mode not in _CATEGORY_MODES:
            allowed = ", ".join(sorted(_CATEGORY_MODES))
            raise ValueError(
                "build.category_mode must be one of: {}.".format(allowed)
            )

        if not 0 <= self.constant_category <= 9:
            raise ValueError("build.constant_category must be between 0 and 9.")

        if self.max_trajectories is not None and self.max_trajectories <= 0:
            raise ValueError(
                "build.max_trajectories must be a positive integer or null."
            )

        if self.epochs <= 0:
            raise ValueError("train.epochs must be positive.")

        if self.batch_size <= 0:
            raise ValueError("train.batch_size must be positive.")

        if self.save_params_rate <= 0:
            raise ValueError("train.save_params_rate must be positive.")

        if self.max_length is not None and self.max_length <= 0:
            raise ValueError("train.max_length must be positive or null.")

        if self.checkpoint_epoch <= 0:
            raise ValueError("generate.checkpoint_epoch must be positive.")

        if self.checkpoint_epoch > self.epochs:
            raise ValueError(
                "generate.checkpoint_epoch cannot be greater than train.epochs."
            )

        if self.checkpoint_epoch % self.save_params_rate != 0:
            raise ValueError(
                "generate.checkpoint_epoch must be divisible by "
                "train.save_params_rate so the requested checkpoint is saved."
            )

    def _validate_source_inputs(self):
        """Validate mutually exclusive combined and pre-split input modes."""
        has_source = self.source_path is not None
        has_train = self.train_source_path is not None
        has_test = self.test_source_path is not None

        if has_source and (has_train or has_test):
            raise ValueError(
                "Configure either paths.source or the "
                "paths.train_source/paths.test_source pair, not both."
            )

        if not has_source and not (has_train and has_test):
            raise ValueError(
                "Configure either paths.source or both "
                "paths.train_source and paths.test_source."
            )

        if has_source:
            self._validate_train_ratio()


    def _validate_train_ratio(self):
        """Validate the split ratio used for a combined source dataset."""
        if self.train_ratio is None:
            raise ValueError(
                "build.train_ratio is required when paths.source is used."
            )

        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError(
                "build.train_ratio must be between 0 and 1."
            )


def load_pipeline_config(config_path):
    """Load a YAML wrapper config and return a validated PipelineConfig."""
    config_path = Path(config_path).expanduser()

    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    if not config_path.is_file():
        raise FileNotFoundError(
            "Pipeline configuration does not exist: {}".format(config_path)
        )

    with config_path.open("r") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise TypeError(
            "Pipeline configuration must contain a YAML mapping at the top level."
        )

    dataset = _require_mapping(raw, "dataset")
    paths = _require_mapping(raw, "paths")
    build = _require_mapping(raw, "build")
    encode = _require_mapping(raw, "encode")
    train = _require_mapping(raw, "train")
    generate = _require_mapping(raw, "generate")

    workspace_dir = _as_path(_require_value(paths, "workspace"))
    repo_dir = _as_path(_require_value(paths, "repo"))
    data_dir = _as_path(_require_value(paths, "data"))
    outputs_dir = _as_path(_require_value(paths, "outputs"))

    source_path = _optional_path(paths, "source")
    train_source_path = _optional_path(paths, "train_source")
    test_source_path = _optional_path(paths, "test_source")
    builder_path = _as_path(_require_value(paths, "builder"))

    checkpoint_epoch = generate.get("checkpoint_epoch")
    if checkpoint_epoch is None:
        checkpoint_epoch = _require_int(train, "epochs")
    else:
        checkpoint_epoch = _positive_int_value(
            checkpoint_epoch,
            "generate.checkpoint_epoch",
        )

    max_length = train.get("max_length")
    if max_length is not None:
        max_length = _positive_int_value(max_length, "train.max_length")

    max_trajectories = build.get("max_trajectories")
    if max_trajectories is not None:
        max_trajectories = _positive_int_value(
            max_trajectories,
            "build.max_trajectories",
        )

    return PipelineConfig(
        dataset_name=_require_str(dataset, "name"),
        workspace_dir=workspace_dir,
        repo_dir=repo_dir,
        data_dir=data_dir,
        outputs_dir=outputs_dir,
        source_path=source_path,
        train_source_path=train_source_path,
        test_source_path=test_source_path,
        builder_path=builder_path,
        train_ratio=_optional_number(build, "train_ratio"),
        random_state=_require_int(build, "random_state"),
        category_mode=_require_str(build, "category_mode").lower(),
        constant_category=_require_int(build, "constant_category"),
        max_trajectories=max_trajectories,
        tid_col=_require_str(encode, "tid_col"),
        epochs=_require_int(train, "epochs"),
        batch_size=_require_int(train, "batch_size"),
        save_params_rate=_require_int(train, "save_params_rate"),
        max_length=max_length,
        checkpoint_epoch=checkpoint_epoch,
    )


def _as_path(value):
    if not isinstance(value, str):
        raise TypeError("Configured paths must be strings.")
    return Path(value).expanduser()

def _optional_path(mapping, key):
    """Read an optional filesystem path from a YAML mapping."""
    value = mapping.get(key)

    if value is None:
        return None

    return _as_path(value)

def _require_mapping(mapping, key):
    value = _require_value(mapping, key)
    if not isinstance(value, dict):
        raise TypeError("'{}' must be a YAML mapping.".format(key))
    return value


def _require_value(mapping, key):
    if key not in mapping:
        raise ValueError(
            "Missing required pipeline configuration key: {}".format(key)
        )
    return mapping[key]


def _require_str(mapping, key):
    value = _require_value(mapping, key)
    if not isinstance(value, str):
        raise TypeError("'{}' must be a string.".format(key))
    value = value.strip()
    if not value:
        raise ValueError("'{}' cannot be empty.".format(key))
    return value


def _require_int(mapping, key):
    value = _require_value(mapping, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("'{}' must be an integer.".format(key))
    return value


def _require_number(mapping, key):
    value = _require_value(mapping, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("'{}' must be a number.".format(key))
    return float(value)

def _optional_number(mapping, key):
    """Read an optional numeric value from a YAML mapping."""
    value = mapping.get(key)

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "'{}' must be a number or null.".format(key)
        )

    return float(value)

def _positive_int_value(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("{} must be an integer or null.".format(label))
    if value <= 0:
        raise ValueError("{} must be positive.".format(label))
    return value
