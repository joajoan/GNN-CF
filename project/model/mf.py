import torch

from torch import Tensor
from torch.nn import Module
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from torch import device

from .base import InnerProduct, NodeEmbedding
from .base import triplet_handler


__all__ = (
    
    # Module.
    'MF',

    # Function.
    'evaluate',
    'predict'
)


class MF(Module):

    def __init__(self, 
            num_embeddings: dict[NodeType, int], 
            embedding_dim: int,
            **kwargs
        ) -> None:
        super().__init__()
        self.embedding = NodeEmbedding(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
            **kwargs
        )
        self.regressor = InnerProduct()
        

    def forward(self, 
        src_n_id: tuple[NodeType, Tensor], 
        dst_n_id: tuple[NodeType, Tensor],
        edge_index: Tensor, 
    ) -> Tensor:
        # Unpacks the input arguments.
        src_node, src_n_id = src_n_id
        dst_node, dst_n_id = dst_n_id
        src_idx, dst_idx = edge_index
        # Constructs the embeddings.
        src_x, dst_x = self.embedding({
            src_node: src_n_id,
            dst_node: dst_n_id
        }).values()
        # Computes and returns the predicted scores.
        return self.regressor(src_x[src_idx], dst_x[dst_idx])
    

def evaluate(
    module: MF, 
    data: HeteroData, 
    loss_fn: callable,
    *,
    edge_type: EdgeType,
    device: device
) -> Tensor:
    # Unpacks the given edge_type.
    src_node, _, dst_node = edge_type
    # Sends the items to the correct device.
    data = data.to(device)

    # Computes the embedding propegation.
    x = module.embedding({
        src_node: data[src_node].n_id,
        dst_node: data[dst_node].n_id
    })
    # Extracts the sought feature tensors.
    src_x, dst_pos_x, dst_neg_x = triplet_handler(data, x,
        src_node=src_node,
        dst_node=dst_node
    )
    # Computes the link scores.
    y_pos = module.regressor(src_x, dst_pos_x)
    y_neg = module.regressor(src_x, dst_neg_x)

    # Computes the loss.
    loss = loss_fn(y_pos, y_neg)

    # Returns the loss.
    return loss


@torch.no_grad()
def predict(
    module: MF, 
    data: HeteroData, 
    *,
    edge_type: EdgeType,
    device: device
) -> Tensor:
    
    # Sends the items to the correct device.
    data = data.to(device)

    # Extracts the edge (label) index.
    try:
        edge_index = data[edge_type].edge_label_index
    except AttributeError:
        edge_index = data[edge_type].edge_index

    # Unpacks the given edge_type.
    src_node, _, dst_node = edge_type
    # Computes the edge scores.
    scores = module(
        src_n_id=(src_node, data[src_node].n_id),
        dst_n_id=(dst_node, data[dst_node].n_id),
        edge_index=edge_index
    )

    # Returns the computed scores.
    return scores