import torch
import torch.nn as nn
from .diff_module import diff_CSDI
from .diff_util import diff_forward_x0_constraint, cal_x0_from_noise_ddpm

from torch_geometric.data import Data
from urban_syn.gnn.encoder import RoadGNNEncoder
from .road_ids import encode_transition_keys

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
        src, dst = graph.edge_index.long() + 1
        road_id_base = graph.num_nodes + 1

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
        if not torch.equal(graph.edge_id.to(self.edge_ids.device), self.edge_ids):
            raise ValueError("Graph road ordering differs from the model mapping.")

        embeddings = self.road_encoder(
            graph.x,
            graph.edge_index,
            getattr(graph, "edge_attr", None),
        )

        padding = embeddings.new_zeros((1, embeddings.size(-1)))

        return torch.cat((padding, embeddings), dim=0)
        
    def forward(
            self,
            graph: Data,
            src_eid_seqs: torch.Tensor,
            src_rate_seqs: torch.Tensor,
        ):
        """
        Compute the original three losses

        src_eid_seqs: road segment sequence of RNTraj
        src_rate_seqs: moving rate of RNTraj

        Parameters
        ----------
        graph : Data
            Road graph containing selected GNN features.
        src_eid_seqs : Tensor
            One-based diffusion road IDs [B, T], with zero padding.
        src_rate_seqs : Tensor
            Position rates [B, T].

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Noise prediction, connectivity, and reconstruction losses.
        """
        # batchsize, max_src_len = src_eid_seqs.shape

        # Recompute embeddings during training to preserve gradient flow
        road_embeddings = self.encode_roads(graph)
        id_embeddings = road_embeddings[src_eid_seqs.long()] # B, T, d
        
        # id_embed = SE[src_eid_seqs]  # B, T, d
        # input_data = torch.cat((id_embed, src_rate_seqs.unsqueeze(-1)), -1)  # B, T, d+1
        input_data = torch.cat((id_embeddings, src_rate_seqs.unsqueeze(-1)), dim=-1) # B, T, d+1
        
        diff_noise, const_loss, x0_loss = diff_forward_x0_constraint(
            self.diff_model,
            input_data,
            self.diffusion_param,
            road_embeddings,
            self.valid_transition_keys,
            road_id_base=self.edge_ids.numel() + 1,
            similarity_chunk_size=self.similarity_chunk_size,
        )
        
        return diff_noise, const_loss, x0_loss

    @torch.no_grad()
    def generate_data(
        self,
        graph: Data,
        batchsize: int,
        length: int,
        similarity_chunk_size: int | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate road-ID sequences and position rates

        Parameters
        ----------
        graph : Data
            Road graph with selected GNN features.
        batchsize : int
            Number of trajectories to generate.
        length : int
            Number of positions per trajectory.
        similarity_chunk_size : int | None, default=None
            Number of positions compared per batch. Uses the model setting
            when not provided.

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
        chunk_size = (
            self.similarity_chunk_size
            if similarity_chunk_size is None
            else similarity_chunk_size
        )

        road_embeddings = self.encode_roads(graph)
        pre_dim = road_embeddings.size(-1)
        
        x0 = cal_x0_from_noise_ddpm(
            self.diff_model,
            self.diffusion_param,
            batchsize,
            length,
            pre_dim + 1
        )  # B, T, pre_dim + 1

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
