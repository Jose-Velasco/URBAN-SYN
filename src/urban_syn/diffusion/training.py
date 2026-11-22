"""Joint training utilities for road-GNN and Diff-RNTraj."""
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import Data

from urban_syn.gnn import RoadGNNEncoder, make_conv_factory
from urban_syn.graph.config import build_dataset_from_config, load_graph_config
from urban_syn.graph.selection import select_graph_features
from urban_syn.graph.validation import validate_graph_features

from .config import JointTrainingSettings, TrainingSettings
from .diff_module import diff_CSDI
from .model import Diff_RNTraj


def build_joint_model(
    graph: Data,
    settings: JointTrainingSettings,
) -> Diff_RNTraj:
    """Build the GNN and diffusion denoiser as one trainable model.

    Parameters
    ----------
    graph : Data
        Road graph with selected node and edge features.
    settings : JointTrainingSettings
        Validated architecture and diffusion settings.

    Returns
    -------
    Diff_RNTraj
        Joint model on the same device as the graph.
    """
    gnn_settings = settings.gnn
    diff_settings = settings.diffusion

    road_encoder = RoadGNNEncoder.from_graph(
        graph,
        gnn_settings,
        make_conv_factory(gnn_settings),
    )

    # Preserve square-root-spaced beta schedule.
    beta = torch.linspace(
        diff_settings.beta_start ** 0.5,
        diff_settings.beta_end ** 0.5,
        diff_settings.diffusion_steps,
    ).square()

    alpha = 1 - beta
    diffusion_hyperparams = {
        "T": diff_settings.diffusion_steps,
        "alpha": alpha,
        "alpha_bar": torch.cumprod(alpha, dim=0),
    }

    # Road embedding width is controlled exclusively by the GNN.
    denoiser = diff_CSDI(
        channels=diff_settings.hidden_dim,
        inputdim=diff_settings.hidden_dim,
        num_steps=diff_settings.diffusion_steps,
        embedding_dim=diff_settings.hidden_dim,
        pre_dim=gnn_settings.output_dim,
        rdcl=diff_settings.residual_layers,
    )

    model = Diff_RNTraj(
        diff_model=denoiser,
        diffusion_hyperparams=diffusion_hyperparams,
        road_encoder=road_encoder,
        graph=graph,
        similarity_chunk_size=diff_settings.similarity_chunk_size,
    )

    return model.to(graph.x.device) # pyright: ignore[reportOptionalMemberAccess]


def load_training_graph(config_path: Path) -> Data:
    """Load the road graph and select configured GNN features.

    Parameters
    ----------
    config_path : Path
        Existing graph YAML configuration.

    Returns
    -------
    Data
        Graph containing the selected GNN inputs and road IDs.
    """
    config = load_graph_config(config_path)
    dataset = build_dataset_from_config(config)
    graph = dataset[0]

    model_config = config["model"]
    node_features = model_config.get("node_features", ["x"])
    edge_features = model_config.get("edge_features", [])

    validate_graph_features(
        graph, # pyright: ignore[reportArgumentType]
        node_features=node_features,
        edge_features=edge_features,
    )

    graph = select_graph_features(
        graph, # pyright: ignore[reportArgumentType]
        node_features=node_features,
        edge_features=edge_features,
    )

    # Keep only attributes needed by the joint model
    # Avoid transferring unused graph features to GPU
    return Data(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=getattr(graph, "edge_attr", None),
        edge_id=graph.edge_id,
    )


def train_one_epoch(
    model: Diff_RNTraj,
    graph: Data,
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    settings: TrainingSettings,
) -> dict[str, float]:
    """Train both networks for one epoch using prepared trajectory batches.

    Parameters
    ----------
    model : Diff_RNTraj
        Joint GNN and diffusion model.
    graph : Data
        Road graph on the model device.
    batches : Iterable[tuple[Tensor, Tensor]]
        Pairs of model road IDs and raw position rates [B, T].
        Road IDs are one-based, with 0 reserved for padding.
        Rates are in [0, 1].
    optimizer : Optimizer
        Optimizer containing both networks' parameters.
    settings : TrainingSettings
        Gradient clipping and optional batch limit.

    Returns
    -------
    dict[str, float]
        Mean total, noise, connectivity, and reconstruction losses.
    """
    model.train()
    totals = {
        "total_loss": 0.0,
        "noise_loss": 0.0,
        "const_loss": 0.0,
        "x0_loss": 0.0,
    }
    num_batches = 0

    for road_ids, rates in batches:
        if (
            settings.max_batches is not None
            and num_batches >= settings.max_batches
        ):
            break

        road_ids = road_ids.to(graph.x.device, dtype=torch.long) # pyright: ignore[reportOptionalMemberAccess]
        rates = rates.to(graph.x.device, dtype=graph.x.dtype) # pyright: ignore[reportOptionalMemberAccess]

        # Preserve original [0, 1] -> [-1, 1] rate transform.
        normalized_rates = 2 * rates - 1

        optimizer.zero_grad(set_to_none=True)

        noise_loss, const_loss, x0_loss = model(
            graph,
            road_ids,
            normalized_rates,
        )

        total_loss = noise_loss + const_loss + x0_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            settings.grad_clip_norm,
        )
        optimizer.step()

        totals["total_loss"] += total_loss.item()
        totals["noise_loss"] += noise_loss.item()
        totals["const_loss"] += const_loss.item()
        totals["x0_loss"] += x0_loss.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Training batches must not be empty.")

    return {
        name: value / num_batches
        for name, value in totals.items()
    }


def save_joint_checkpoint(
    path: str | Path,
    model: Diff_RNTraj,
    optimizer: torch.optim.Optimizer,
    settings: JointTrainingSettings,
    epoch: int,
) -> None:
    """Save model weights, optimizer state, and reproducible settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": settings.model_dump(mode="json"),
        },
        path,
    )
