"""Generate GPS trajectories from a trained joint GNN + diffusion checkpoint.

Encode the graph once, then release the graph and encoder from this inference
model. All denoseing batches reuse the cached road embeddings.

example usage:

uv run python -m urban_syn.diffusion.generate \
  --config configs/graph/nyc_graph.yaml \
  --checkpoint outputs/checkpoints/joint_diffusion_synthetic.pt \
  --road-network fmm_scripts/data/road_network/nyc.gpkg \
  --road-layer roads \
  --length-distribution fmm_scripts/outputs/urban_syn_ready/nyc_15s_enriched_moves_train_length_distribution.npy \
  --num-trajectories 8 \
  --batch-size 2 \
  --seed 101 \
  --output outputs/urban_syn/generated/nyc_generated_gps.parquet \
  --log-file outputs/logs/nyc_generation.log
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from urban_syn.experiment.factories import build_sampler
from .config import JointTrainingSettings, load_joint_training_config
from .reconstruction import load_road_geometries, reconstruct_gps_trajectory
from .road_ids import encode_transition_keys, model_ids_to_edge_ids
from .training import build_joint_model, load_training_graph
from .sampling.base import Sampler

logger = logging.getLogger(__name__)

_RAW_RATE_BINS = np.array([
    -np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf
])

_DECODED_RATE_BINS = np.linspace(0.0, 1.0, 11)

@dataclass
class RateDiagnostics:
    """Accumulate generation-rate statistics without storing all samples."""

    raw_hist: np.ndarray = field(
        default_factory=lambda: np.zeros(len(_RAW_RATE_BINS) - 1, dtype=np.int64)
    )
    decoded_hist: np.ndarray = field(
        default_factory=lambda: np.zeros(len(_DECODED_RATE_BINS) - 1, dtype=np.int64)
    )

    clipped_low: int = 0
    clipped_high: int = 0
    endpoint_count: int = 0
    raw_min: float = float("inf")
    raw_max: float = float("-inf")

    def update(self, raw_rates: Tensor, decoded_rates: Tensor) -> None:
        """Collect statistics before and after clamping/rescaling."""
        raw = raw_rates.detach().cpu().numpy()
        decoded = decoded_rates.detach().cpu().numpy()

        if not np.isfinite(raw).all() or not np.isfinite(decoded).all():
            raise ValueError("Generated rates contain non-finite values.")

        if raw.size == 0:
            return

        self.raw_hist += np.histogram(raw, bins=_RAW_RATE_BINS)[0]
        self.decoded_hist += np.histogram(decoded, bins=_DECODED_RATE_BINS)[0]

        self.clipped_low += int(np.count_nonzero(raw < -1.0))
        self.clipped_high += int(np.count_nonzero(raw > 1.0))
        self.endpoint_count += int(
            np.count_nonzero((decoded == 0.0) | (decoded == 1.0))
        )

        self.raw_min = min(self.raw_min, float(raw.min()))
        self.raw_max = max(self.raw_max, float(raw.max()))

    def log_summary(self) -> None:
        """Log aggregate statistics across all generation batches."""
        total = int(self.raw_hist.sum())
        if total == 0:
            logger.info("No rate samples generated.")
            return

        clipped = self.clipped_low + self.clipped_high

        logger.info(
            "Raw rates: min=%.4f, max=%.4f",
            self.raw_min, self.raw_max,
        )
        logger.info(
            "Raw histogram: bins=%s, counts=%s",
            _RAW_RATE_BINS.tolist(), self.raw_hist.tolist(),
        )
        logger.info(
            "Rates clipped: %d/%d (%.2f%%), below=%d, above=%d",
            clipped, total, 100.0 * clipped / total,
            self.clipped_low, self.clipped_high,
        )
        logger.info(
            "Decoded [0, 1] histogram: %s",
            self.decoded_hist.tolist(),
        )
        logger.info(
            "Decoded endpoint rates (0 or 1): %d/%d (%.2f%%)",
            self.endpoint_count, total,
            100.0 * self.endpoint_count / total,
        )

def parse_args() -> argparse.Namespace:
    """Read generation inputs, sampling options, and output paths."""
    parser = argparse.ArgumentParser(
        description="Generate GPS trajectories from a trained joint GNN + diffusion checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=Path, required=True,
        help="Graph/model/training YAML used to construct the trained model and selected PyG graph.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the saved joint GNN + diffusion model checkpoint (.pt).",
    )
    parser.add_argument(
        "--road-network", type=Path, required=True,
        help="Shared road-network GeoPackage (.gpkg) with full road geometries and edge_id values.",
    )
    parser.add_argument(
        "--road-layer", default="roads",
        help="Layer containing the road geometries within --road-network.",
    )
    parser.add_argument(
        "--length-distribution", type=Path, required=True,
        help="Training sequence-length counts (.npy), used to choose generated trajectory lengths.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Destination Parquet path; one row per generated GPS observation.",
    )
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="Optional log-file path; logs also appear in the terminal. Parent folders are created.",
    )
    parser.add_argument(
        "--num-trajectories", type=int, default=None,
        help="Number of trajectories to sample. If omitted, uses the exact training length counts.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Number of trajectories generated per diffusion batch; reduce to limit GPU memory.",
    )
    parser.add_argument(
        "--seed", type=int, default=101,
        help="Random seed for trajectory-length selection and diffusion sampling.",
    )
    parser.add_argument(
        "--similarity-chunk-size", type=int, default=None,
        help="Positions decoded per cosine-similarity chunk; defaults to the model setting.",
    )

    return parser.parse_args()


def setup_logging(log_file: Path | None) -> None:
    """Log to the terminal and optionally append to a specified file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )

def _validate_on_select_generation_lengths(counts) -> None:
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("Length distribution must be a 1-D integer count array.")
    if (counts < 0).any() or len(counts) < 2 or counts[1:].sum() <= 0:
        raise ValueError("Length distribution has no valid positive-length trajectories.")
    if counts[0] != 0:
        raise ValueError("Length distribution cannot contain zero-length trajectories.")

def _validate_on_generate_gps_trajectories(batch_size: int, road_vocab: Tensor, road_embeddings: Tensor) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if road_vocab.ndim != 1 or road_embeddings.ndim != 2:
        raise ValueError("Expected 1-D road_vocab and 2-D road_embeddings.")
    if road_embeddings.size(0) != road_vocab.numel() + 1:
        raise ValueError("Road embeddings and road vocabulary have different sizes.")

def _validate_generate_gps_trajectories_post_denoise_sampling(
        model_ids: Tensor,
        batch_count: int,
        length: int,
        normalized_rates: Tensor,
        road_vocab: Tensor
        ) -> None:
    if model_ids.shape != (batch_count, length) or normalized_rates.shape != model_ids.shape:
        raise ValueError("Diffusion sampler returned inconsistent [batch, length] shapes.")
    
    if ((model_ids < 1) | (model_ids > road_vocab.numel())).any():
        raise ValueError("Sampler returned padding or an invalid model road ID.")


def _checkpoint_model_state(checkpoint: Mapping) -> Mapping[str, Tensor]:
    """Extract model weights from an existing joint-training checkpoint."""
    for name in ("model_state_dict", "model", "state_dict"):
        state = checkpoint.get(name)
        if isinstance(state, Mapping):
            return state
    
    raise ValueError("Checkpoint is missing model weights (model_state_dict/model/state_dict).")


def _check_graph_ids(model_state: Mapping[str, Tensor], graph: Data) -> None:
    """Refuse checkpoint loading if the road vocabulary or node order differs."""
    saved_ids = model_state.get("edge_ids")
    if not isinstance(saved_ids, Tensor):
        raise ValueError(
            "Checkpoint has no saved edge_ids buffer; cannot verify road-ID ordering."
        )
    
    current_ids = graph.edge_id.detach().cpu().long()
    if not torch.equal(saved_ids.detach().cpu().long(), current_ids):
        raise ValueError(
            "Checkpoint and graph have different shared road-network edge IDs "
            "or PyG node ordering. Load the graph used for training."
        )

    saved_transitions = model_state.get("valid_transition_keys")
    if not isinstance(saved_transitions, Tensor):
        raise ValueError("Checkpoint has no saved valid_transition_keys buffer.")

    # Compare directed road connectivity too. Otherwise loading the checkpoint
    # could silently overwrite the graph's newly built connectivity keys.
    src, dst = graph.edge_index.detach().cpu().long() + 1 # pyright: ignore[reportOptionalMemberAccess]
    current_transitions = encode_transition_keys(src, dst, len(current_ids) + 1)
    current_transitions = torch.unique(current_transitions).sort().values
    saved_transitions = torch.unique(saved_transitions.detach().cpu().long()).sort().values
    
    if not torch.equal(saved_transitions, current_transitions):
        raise ValueError("Checkpoint and graph have different directed road transitions.")


def load_generation_model(
    checkpoint_path: str | Path,
    graph: Data,
    settings: JointTrainingSettings,
    device: torch.device,
) -> nn.Module:
    """Restore jointly trained GNN and diffusion weights using the selected graph.

    Args:
        checkpoint_path: Existing joint-training checkpoint created by training.py.
        graph: Feature-selected PyG graph from the training network.
        settings: Same model configuration used for training.
        device: Target device for model and graph (currently CUDA for sampling).
    """
    # Load only a locally trusted checkpoint. The current training checkpoint
    # contains tensor weights and serializable config/optimizer dictionaries.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state = _checkpoint_model_state(checkpoint)
    _check_graph_ids(model_state, graph)

    model = build_joint_model(graph, settings).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model


def select_generation_lengths(
    distribution_path: str | Path,
    num_trajectories: int | None = None,
    seed: int = 42,
) -> dict[int, int]:
    """Choose generation counts per length from the training distribution.

    If num_trajectories is omitted, reproduce the training length counts.
    Otherwise sample a fixed total according to their relative frequencies.
    """
    counts = np.load(distribution_path, allow_pickle=False)
    _validate_on_select_generation_lengths(counts)

    lengths = np.flatnonzero(counts[1:]) + 1
    observed_counts = counts[lengths].astype(np.int64)
    
    if num_trajectories is None:
        sampled_counts = observed_counts
    else:
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be positive.")
    
        probabilities = observed_counts / observed_counts.sum()
        sampled_counts = np.random.default_rng(seed).multinomial(num_trajectories, probabilities)

    return {
        int(length): int(count)
        for length, count in zip(lengths, sampled_counts)
        if count > 0
    }

def _trajectory_to_records(
    trajectory_id: int,
    model_sequence: Tensor,
    edge_sequence: Tensor,
    rates: Sequence[float],
    road_geometries: Mapping,
) -> list[dict]:
    """Reconstruct one generated trajectory and format its GPS records."""
    model_ids = model_sequence.tolist()
    edge_ids = edge_sequence.tolist()

    gps_points = reconstruct_gps_trajectory(
        edge_ids, rates, road_geometries
    )

    result = [
        {
            "trajectory_id": trajectory_id,
            "step": step,
            "model_road_id": model_id,
            "edge_id": edge_id,
            "rate": rate,
            "position_rate": position_rate,
            "lng": lon,
            "lat": lat,
        }
        for step, (model_id, edge_id, rate, (lon, lat, position_rate))
        in enumerate(zip(model_ids, edge_ids, rates, gps_points, strict=True))
    ]
    return result

@torch.inference_mode()
def generate_gps_trajectories(
    model: nn.Module,
    road_vocab: Tensor,
    road_embeddings: Tensor,
    road_geometries: Mapping,
    length_counts: Mapping[int, int],
    sampler: Sampler,
    batch_size: int = 2,
    similarity_chunk_size: int | None = None,
) -> pd.DataFrame:
    """Generate GPS trajectories using an injectable diffusion sampler.

    Parameters
    ----------
    model : nn.Module
        Trained joint GNN and diffusion model.
    road_vocab : Tensor
        Physical road IDs in PyG node order, shape [N].
    road_embeddings : Tensor
        Precomputed road embeddings, including padding, shape [N + 1, D].
    road_geometries : Mapping
        Road geometries used to reconstruct GPS coordinates.
    length_counts : Mapping[int, int]
        Number of trajectories to generate for each sequence length.
    batch_size : int, default=2
        Maximum number of trajectories generated per batch.
    similarity_chunk_size : int | None, default=None
        Number of trajectory positions decoded per similarity chunk.
    sampler : Sampler
        Sampling strategy injected into the model's generation method.

    Returns
    -------
    pd.DataFrame
        Generated GPS observations with trajectory IDs, road IDs,
        normalized and decoded rates, and geographic coordinates.
    """
    _validate_on_generate_gps_trajectories(batch_size, road_vocab, road_embeddings)

    model.eval()
    road_vocab = road_vocab.detach().cpu().long()
    records: list[dict] = []
    trajectory_id = 0
    # Track rates distributions across all generation batches for sampling diagnostics and visibility.
    rate_diagnostics = RateDiagnostics()

    for length, count in sorted(length_counts.items()):
        if length < 1 or count < 0:
            raise ValueError(f"Invalid generation length/count: {length}, {count}.")
        
        logger.info(f"Generating {count:,} trajectories of length {length}.")
        for start in range(0, count, batch_size):
            batch_count = min(batch_size, count - start)

            # denoseing uses the cached road table: no GNN or graph per batch.
            model_ids, normalized_rates = model.generate_data(
                graph=None,
                batchsize=batch_count,
                length=length,
                sampler=sampler,
                similarity_chunk_size=similarity_chunk_size,
                road_embeddings=road_embeddings
            ) # pyright: ignore[reportCallIssue]

            model_ids = torch.as_tensor(model_ids).detach().cpu().long()
            normalized_rates = torch.as_tensor(normalized_rates).detach().cpu()
            
            _validate_generate_gps_trajectories_post_denoise_sampling(
                model_ids=model_ids,
                batch_count=batch_count,
                length=length,
                normalized_rates=normalized_rates,
                road_vocab=road_vocab
            )

            # Diffusion predicts [-1, 1] rates. The training converter
            # originally encoded rates in [0, 1].
            # Decode normalized predictions and collect rate statistics.
            decoded_rates = (normalized_rates.clamp(-1, 1) + 1) / 2
            rate_diagnostics.update(normalized_rates, decoded_rates)
            rates = decoded_rates.tolist()

            edge_ids = model_ids_to_edge_ids(model_ids, road_vocab)
            edge_ids = torch.as_tensor(edge_ids).cpu().long()

            # Reconstruct GPS observations for each generated trajectory.
            for model_sequence, edge_sequence, rate_sequence in zip(model_ids, edge_ids, rates, strict=True):
                records.extend(
                    _trajectory_to_records(
                        trajectory_id,
                        model_sequence,
                        edge_sequence,
                        rate_sequence,
                        road_geometries,
                    )
                )
                trajectory_id += 1

    rate_diagnostics.log_summary()

    return pd.DataFrame.from_records(
        records,
        columns=[
            "trajectory_id", "step", "model_road_id", "edge_id",
            "rate", "position_rate", "lng", "lat",
        ],
    )


def main() -> None:
    """Reload the joint checkpoint and export generated GPS trajectories."""
    args = parse_args()
    setup_logging(args.log_file)

    logger.info(f"Generation checkpoint: {args.checkpoint}")
    logger.info(f"Generation configuration: {args.config}")

    if not torch.cuda.is_available():
        raise RuntimeError("The existing denoising sampling implementation requires CUDA.")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)


    graph = load_training_graph(args.config).to(device)
    settings = load_joint_training_config(args.config)
    model = load_generation_model(args.checkpoint, graph, settings, device)

    # Preserve the shared road-network IDs before releasing the PyG graph.
    road_vocab = graph.edge_id.detach().cpu().long().clone()
    model.eval()
    with torch.inference_mode():
        # Encode all roads once, not once per generated batch.
        road_embeddings = model.encode_roads(graph) # pyright: ignore[reportCallIssue]

    logger.info(f"Cached road embeddings: {tuple(road_embeddings.shape)}")

    # The checkpoint has already been restored and the road embeddings cached.
    # Neither the PyG graph nor the GNN is used during diffusion sampling.
    del graph
    del model.road_encoder

    length_counts = select_generation_lengths(
        args.length_distribution,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
    )
    roads = load_road_geometries(args.road_network, layer=args.road_layer)
    missing_roads = set(road_vocab.tolist()) - set(roads)
    if missing_roads:
        raise ValueError(f"GeoPackage lacks {len(missing_roads):,} roads present in the PyG graph.")

    sampler = build_sampler("ddpm")

    generated = generate_gps_trajectories(
        model,
        road_vocab,
        road_embeddings,
        roads,
        length_counts,
        sampler=sampler,
        batch_size=args.batch_size,
        similarity_chunk_size=args.similarity_chunk_size
    )

    # generated = generate_gps_trajectories(
    #     model, road_vocab, road_embeddings, roads, length_counts,
    #     batch_size=args.batch_size,
    #     similarity_chunk_size=args.similarity_chunk_size,
    # )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated.to_parquet(args.output, index=False)

    logger.info(
        f"Wrote {generated['trajectory_id'].nunique():,} trajectories "
        f"({len(generated):,} GPS observations) to {args.output}."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Trajectory generation failed.")
        raise
