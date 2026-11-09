"""Run configurable joint GNN and Diff-RNTraj training.

Example
-------
uv run python -m urban_syn.diffusion \
    --config configs/graph/nyc_graph.yaml \
    --checkpoint outputs/our_model/checkpoints/joint_diffusion_synthetic.pt \
    --device cuda \
    --num-batches 2 \
    --trajectory-length 8

^^^ old version that use random generated data

uv run python -m urban_syn.diffusion \
  --config configs/graph/nyc_graph.yaml \
  --train-parquet ./fmm_scripts/outputs/urban_syn_ready/nyc_15s_enriched_moves_train_prepared_road_sequences.parquet
"""
import argparse
import logging
from pathlib import Path
import torch

from urban_syn.graph.utils import graph_info

from .data import RoadSequenceDataset, make_length_dataloader

from .config import load_joint_training_config
from .training import (
    build_joint_model,
    load_training_graph,
    save_joint_checkpoint,
    train_one_epoch,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse configuration and synthetic training workload options."""
    parser = argparse.ArgumentParser(
        description="Train the joint GNN and Diff-RNTraj model."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument("--seed", type=int, default=101, help="Random seed for reproducible batch shuffling.",)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/joint_diffusion_synthetic.pt"),
    )

    parser.add_argument(
        "--train-parquet",
        type=Path,
        required=True,
        help="Prepared training trajectories containing road edge_ids and rates.",
    )

    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    """Select an available training device."""
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    use_cuda = value == "cuda" or (
        value == "auto" and torch.cuda.is_available()
    )
    return torch.device("cuda" if use_cuda else "cpu")

def main() -> None:
    """Build the joint model and run configurable synthetic training."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    logger.info("Loading configuration: %s", args.config)
    settings = load_joint_training_config(args.config)

    # Reuse the existing processed graph and selected feature groups
    graph = load_training_graph(args.config).to(device)

    logger.info("\n%s", graph_info(graph))

    logger.info("Device: %s", device)

    model = build_joint_model(graph, settings)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )

    logger.info(
        "GNN: %s | Embedding dimension: %d",
        settings.gnn.architecture,
        settings.gnn.output_dim,
    )
    logger.info(
        "Trainable parameters: %s",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # Generate batches on CPU 
    # generator = torch.Generator().manual_seed(args.seed)

    # Load real map-matched trajectories and convert shared road edge_ids
    # to the model's road IDs using the current PyG graph.
    train_dataset = RoadSequenceDataset(
        args.train_parquet,
        graph.edge_id,
    )

    # Group trajectories by length to produce [B, T] batches without padding.
    # Rates remain in [0, 1] until normalized by train_one_epoch().
    train_loader = make_length_dataloader(
        train_dataset,
        batch_size=settings.training.batch_size,
        shuffle=True,
        seed=args.seed,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"Loaded {len(train_dataset):,} real training trajectories "
        f"across {len(train_loader):,} length-based batches."
    )

    for epoch in range(1, settings.training.epochs + 1):
        metrics = train_one_epoch(
            model=model,
            graph=graph,
            batches=train_loader,
            optimizer=optimizer,
            settings=settings.training,
        )

        logger.info(
            "Epoch %d/%d | Total: %.4f | Noise: %.4f | "
            "Connectivity: %.4f | Reconstruction: %.4f",
            epoch,
            settings.training.epochs,
            metrics["total_loss"],
            metrics["noise_loss"],
            metrics["const_loss"],
            metrics["x0_loss"],
        )

    save_joint_checkpoint(
        path=args.checkpoint,
        model=model,
        optimizer=optimizer,
        settings=settings,
        epoch=settings.training.epochs,
    )

    logger.info("Checkpoint saved: %s", args.checkpoint)

    if device.type == "cuda":
        logger.info(
            "Peak CUDA allocated memory: %.1f MiB",
            torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        )
        logger.info(
            "Peak CUDA reserved memory: %.1f MiB",
            torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        )

if __name__ == "__main__":
    main()
