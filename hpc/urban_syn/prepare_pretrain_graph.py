"""Reuse the existing joint graph preparation for link pretraining on HPC."""

from __future__ import annotations

import argparse
from pathlib import Path

from urban_syn.experiment.graph_preparation import prepare_experiment_graph
from urban_syn.link_pretrain.config import load_link_pretrain_config


def main() -> None:
    """Prepare missing graph-cache artifacts without changing model settings.

    Both resolved configurations expose the same graph YAML and project root.
    The existing graph builder already owns cache validation and preprocessing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    config = load_link_pretrain_config(args.config, project_root=args.project_root)
    prepare_experiment_graph(config)


if __name__ == "__main__":
    main()
