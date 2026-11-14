import torch
import torch.nn as nn
# from .diff_util import diff_forward_x0_constraint, cal_x0_from_noise_ddpm
from .diff_util import (
    DiffusionForwardOutput,
    diffuse_and_denoise
)
from dataclasses import dataclass
from torch_geometric.data import Data
from urban_syn.gnn.encoder import RoadGNNEncoder
from .road_ids import encode_transition_keys
from collections.abc import Mapping
from .losses import (
    LossComposer,
    LossContext,
    LossResult,
    original_loss_composer
)
from .sampling.base import Sampler

@dataclass(frozen=True)
class JointTrainingOutput:
    """Outputs shared by diffusion objectives and road-link losses.

    Attributes
    ----------
    diffusion : DiffusionForwardOutput
        Intermediate tensors from forward noising and denoising.
    road_embeddings : torch.Tensor
        GNN-generated road embeddings, including the padding row.
        Retains autograd history for joint training.
    """

    diffusion: DiffusionForwardOutput
    road_embeddings: torch.Tensor

class Diff_RNTraj(nn.Module):
    def __init__(
            self,
            diff_model: nn.Module,
            diffusion_hyperparams: dict,
            road_encoder: RoadGNNEncoder,
            graph: Data,
            similarity_chunk_size: int = 128
        ):
        """Initialize the diffusion model with a trainable GNN road encoder.

        Parameters
        ----------
        diff_model : nn.Module
            Diffusion denoiser.
        diffusion_hyperparams : dict
            Diffusion schedule containing T, alpha, and alpha_bar.
        road_encoder : RoadGNNEncoder
            Trainable road representation model.
        graph : Data
            Physical road graph used to establish road IDs and valid transitions.
        similarity_chunk_size : int, default=128
            Number of predicted road vectors processed per similarity batch.
        """
        super(Diff_RNTraj, self).__init__()
        if diff_model.pre_dim != road_encoder.settings.output_dim:
            raise ValueError(
                "Diffusion road dimension must match the GNN output dimension."
            )

        if torch.unique(graph.edge_id).numel() != graph.num_nodes:
            raise ValueError("Graph edge_id values must be unique.")

        self.diff_model = diff_model
        self.road_encoder = road_encoder
        self.diffusion_steps = int(diffusion_hyperparams["T"])
        self.similarity_chunk_size = similarity_chunk_size

        # Store the schedule as buffers so model.to(device) moves it with
        # the GNN and denoiser, and checkpoints preserve it.
        self.register_buffer(
            "alpha",
            torch.as_tensor(diffusion_hyperparams["alpha"]).float().clone(),
        )
        self.register_buffer(
            "alpha_bar",
            torch.as_tensor(diffusion_hyperparams["alpha_bar"]).float().clone(),
        )

        # Diffusion model ID i + 1 corresponds to graph.edge_id[i].
        # Model ID 0 is reserved for padding.
        self.register_buffer("edge_ids", graph.edge_id.long().clone())

        # Represent each directed physical transition as an integer key.
        # These use the same one-based model IDs as the embedding table.
        src, dst = graph.edge_index.long() + 1 # pyright: ignore[reportOptionalMemberAccess]
        road_id_base = graph.num_nodes + 1 # pyright: ignore[reportOptionalOperand]

        self.register_buffer(
            "valid_transition_keys",
            torch.unique(encode_transition_keys(src, dst, road_id_base)),
        )

        # self.diffusion_param = diffusion_hyperparams

    @property
    def diffusion_param(self) -> dict:
        """Return diffusion parameters using the current device buffers."""
        return {
            "T": self.diffusion_steps,
            "alpha": self.alpha,
            "alpha_bar": self.alpha_bar,
        }

    def encode_roads(self, graph: Data) -> torch.Tensor:
        """Generate embeddings for every road, including zero-padding.

        Parameters
        ----------
        graph : Data
            PyG road graph with selected node and edge features.

        Returns
        -------
        Tensor
            Road embedding table [N + 1, D].
        """
        if not torch.equal(graph.edge_id.to(self.edge_ids.device), self.edge_ids): # pyright: ignore[reportArgumentType]
            raise ValueError("Graph road ordering differs from the model mapping.")

        embeddings = self.road_encoder(
            graph.x,
            graph.edge_index,
            getattr(graph, "edge_attr", None),
        )

        padding = embeddings.new_zeros((1, embeddings.size(-1)))

        return torch.cat((padding, embeddings), dim=0)
        
    def compute_training_outputs(
        self,
        graph: Data,
        src_eid_seqs: torch.Tensor,
        src_rate_seqs: torch.Tensor
    ) -> JointTrainingOutput:
        """Encode roads and compute diffusion outputs without calculating losses.

        Parameters
        ----------
        graph : Data
            Road graph containing the GNN's selected features.
        src_eid_seqs : Tensor
            Model road IDs of shape [B, T], with zero reserved for padding.
        src_rate_seqs : Tensor
            Position rates of shape [B, T], normalized to [-1, 1].

        Returns
        -------
        JointTrainingOutput
            Diffusion intermediate tensors and trainable road embeddings.
            The road embeddings can be reused by additional objectives.
        """
        road_embeddings = self.encode_roads(graph)
        embedded_roads = road_embeddings[src_eid_seqs.long()]

        clean = torch.cat(
            (embedded_roads, src_rate_seqs.unsqueeze(-1)),
            dim=-1
        )

        diffusion = diffuse_and_denoise(
            self.diff_model,
            clean,
            self.diffusion_param
        )

        return JointTrainingOutput(
            diffusion=diffusion,
            road_embeddings=road_embeddings
        )

    def compute_training_losses(
        self,
        output: JointTrainingOutput,
        composer: LossComposer,
        step: int = 0,
        batch: Mapping[str, torch.Tensor] | None = None
    ) -> LossResult:
        """Evaluate independently configured objectives on shared model outputs.

        Parameters
        ----------
        output : JointTrainingOutput
            Output from compute_training_outputs for the current batch.
        composer : LossComposer
            Injected loss implementations and their combination policy.
        step : int, default=0
            Zero-based optimizer step controlling metric-only frequency.
        batch : Mapping[str, Tensor] | None, default=None
            Optional auxiliary tensors for additional objectives.

        Returns
        -------
        LossResult
            Weighted total, gradient-bearing total, and component values.
        """
        context = LossContext(
            diffusion=output.diffusion,
            road_embeddings=output.road_embeddings,
            valid_transition_keys=self.valid_transition_keys,
            road_id_base=self.edge_ids.numel() + 1,
            similarity_chunk_size=self.similarity_chunk_size,
            batch={} if batch is None else batch
        )

        return composer(context, step=step)

    def forward(
        self,
        graph: Data,
        src_eid_seqs: torch.Tensor,
        src_rate_seqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preserve the original three-loss training interface.

        Parameters
        ----------
        graph : Data
            Road graph containing the GNN's selected features.
        src_eid_seqs : Tensor
            Model road IDs of shape [B, T].
        src_rate_seqs : Tensor
            Normalized position rates of shape [B, T].

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Noise MSE, hard connectivity count, and reconstruction MSE.
            The connectivity count remains non-differentiable.
        """
        output = self.compute_training_outputs(
            graph,
            src_eid_seqs,
            src_rate_seqs
        )

        losses = self.compute_training_losses(
            output,
            original_loss_composer()
        )

        return (
            losses.values["noise_mse"],
            losses.values["hard_connectivity"],
            losses.values["reconstruction_mse"]
        )


    @torch.no_grad()
    def generate_data(
        self,
        graph: Data | None,
        batchsize: int,
        length: int,
        sampler: Sampler,
        similarity_chunk_size: int | None = None,
        road_embeddings: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate road-ID sequences and position rates

        Parameters
        ----------
        graph : Data | None
            Road graph with selected GNN features. Used to compute
            road embeddings when road_embeddings is not provided.
            Can be None when using precomputed embeddings.
        batchsize : int
            Number of trajectories to generate.
        length : int
            Number of positions per trajectory.
        similarity_chunk_size : int | None, default=None
            Number of positions compared per batch. Uses the model setting
            when not provided.
        
        road_embeddings : torch.Tensor | None, default=None
            Precomputed embeddings from encode_roads(graph), shaped
            [num_roads + 1, embedding_dim], including padding row 0.
            When provided, skips GNN computation and allows graph=None.
            Must match the model's road vocabulary and device.

        sampler : Sampler
            diffusion sampling strategy.

        Returns
        -------
        tuple[Tensor, Tensor]
            Generated model road IDs (not raw FMM edge_id values) and position rates [B, T]

            Road IDs index the GNN-generated embedding table, not the
            original FMM edge_id values. Model ID 0 is reserved for
            padding and is excluded from generation.

            Convert generated IDs back to FMM edge_id values using
            model_ids_to_edge_ids() before exporting trajectories.
        """
        # Reuse precomputed embeddings, or compute them from the graph.
        if road_embeddings is None:
            if graph is None:
                raise ValueError(
                    "Either graph or road_embeddings must be provided."
                )

            road_embeddings = self.encode_roads(graph)

        chunk_size = (
            self.similarity_chunk_size
            if similarity_chunk_size is None
            else similarity_chunk_size
        )

        pre_dim = road_embeddings.size(-1)

        # selected_sampler = sampler if sampler is not None else DDPMSampler()

        x0 = sampler.sample(
            denoiser=self.diff_model,
            schedule=self.diffusion_param,
            batch_size=batchsize,
            sequence_length=length,
            feature_dim=pre_dim + 1
        )

        # x0 = cal_x0_from_noise_ddpm(
        #     self.diff_model,
        #     self.diffusion_param,
        #     batchsize,
        #     length,
        #     pre_dim + 1
        # )  # B, T, pre_dim + 1

        # Separate the reconstructed road vectors from position rates
        x0_road = x0[..., :pre_dim]  # B, T, pre_dim + 1
        B, T, F = x0_road.shape

        x0_road_shape = x0_road.reshape(B*T, F)

        # Exclude padding ID 0 from road candidates
        candidates = road_embeddings[1:]

        x0_abs = x0_road_shape.norm(dim=1)
        road_abs = candidates.norm(dim=1)

        # sim_matrix = torch.einsum('ik,jk->ij', x0_road_shape, SE) / (torch.einsum('i,j->ij', x0_abs, SE_abs) + 1e-6)
        #
        # sim_matrix = sim_matrix.reshape(B, T, -1)  # B, T, road num    2000 * 20 = 40000 top1->top10
        # sim_matrix = sim_matrix.argmax(-1)
        # Change for Diff-RNTraj (NYC): same cosine similarity and argmax, computed 2,048 points
        # at a time. The full (B*T) x N matrix is ~4 GB for 256 x 100 points and 41,494 roads (and needs
        # ~3 copies), which ran out of memory on a 12 GB P100.
        best = []
        for start in range(0, x0_road_shape.shape[0], chunk_size):
            x_part, x_abs_part = x0_road_shape[start:start + chunk_size], x0_abs[start:start + chunk_size]
            sim = torch.einsum('ik,jk->ij', x_part, candidates) / (torch.einsum('i,j->ij', x_abs_part, road_abs) + 1e-6)
            
            # Convert candidate indices to one-based model road IDs.
            best.append(sim.argmax(-1) + 1)

        sim_matrix = torch.cat(best).reshape(B, T)
        
        rates = x0[:,:,pre_dim]

        return sim_matrix, rates
