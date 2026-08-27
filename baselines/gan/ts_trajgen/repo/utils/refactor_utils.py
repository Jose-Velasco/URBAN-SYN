from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a TS-TrajGen YAML experiment configuration."""
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)