import torch
import torch_geometric as pyg

from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from .base import NodeEmbedding


__all__ = (
    'get_embeddings',
    'get_triplet_indices',
    'get_triplet_xs',
    'make_weight',
    'make_undirected'
)


def get_triplet_indices(
    data: HeteroData, 
    *, 
    src_node: NodeType, 
    dst_node: NodeType
) -> tuple[Tensor, Tensor, Tensor]:
    src_idx = data[src_node].src_index
    dst_pos_idx = data[dst_node].dst_pos_index
    dst_neg_idx = data[dst_node].dst_neg_index
    return src_idx, dst_pos_idx, dst_neg_idx


def get_triplet_xs(
    data: HeteroData, 
    x: dict[NodeType, Tensor],
    *,
    src_node: NodeType,
    dst_node: NodeType
) -> tuple[Tensor, Tensor, Tensor]:
    # Extracts the source and destination nodes' indices.
    src_idx, dst_pos_idx, dst_neg_idx = get_triplet_indices(data,
        src_node=src_node,
        dst_node=dst_node
    )
    # Constructs the nodes' feature matrices.
    src_x = x[src_node][src_idx]
    dst_pos_x = x[dst_node][dst_pos_idx]
    dst_neg_x = x[dst_node][dst_neg_idx]
    # Returns the source, positive and negative features.
    return src_x, dst_pos_x, dst_neg_x


def get_embeddings(
    embedding: NodeEmbedding,
    data: HeteroData, 
    *, 
    src_node: NodeType, 
    dst_node: NodeType
) -> tuple[Tensor, Tensor, Tensor]:
    src_emb, dst_emb = embedding({
        src_node: data[src_node].n_id,
        dst_node: data[dst_node].n_id
    }).values()
    return src_emb, dst_emb


def make_weight(edge_index: Tensor) -> Tensor:
    src_idx, dst_idx = edge_index
    return (
        pyg.utils.degree(src_idx)[src_idx]
        *
        pyg.utils.degree(dst_idx)[dst_idx]
    ).pow(-.5).unsqueeze(-1)


def make_undirected(
    src_edge_index: Tensor, 
    dst_edge_index: Tensor
) -> Tensor:
    return torch.hstack([
        src_edge_index, 
        dst_edge_index.flip(0)
    ]).unique(sorted=False, dim=-1)