import torch

from torch import Tensor
from torch.nn import Module
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from .base import InnerProduct, NodeEmbedding
from torch import device
from . import utils


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
        usr_n_id: tuple[NodeType, Tensor], 
        itm_n_id: tuple[NodeType, Tensor],
        edge_label_index: Tensor
    ) -> Tensor:
        # Unpacks the input arguments.
        usr_node, usr_n_id = usr_n_id
        itm_node, itm_n_id = itm_n_id
        usr_idx, itm_idx = edge_label_index
        # Constructs the embeddings.
        usr_x, itm_x = self.embedding({
            usr_node: usr_n_id,
            itm_node: itm_n_id
        }).values()
        # Computes edge scores.
        edge_score = self.regressor(
            src_x=usr_x[usr_idx], 
            dst_x=itm_x[itm_idx]
        )
        # Returns the estimated edge scores.
        return edge_score
    

def eval_triplet(
    module: MF,
    data: HeteroData, 
    loss_fn: callable,
    *,
    edge_type: EdgeType,
    device: device
) -> Tensor:
    # Unpacks the given edge_type.
    usr_node, _, itm_node = edge_type
    # Sends the items to the correct device.
    data = data.to(device)
    # Computes the embedding propegation.
    x = module.embedding({
        usr_node: data[usr_node].n_id,
        itm_node: data[itm_node].n_id
    })
    # Extracts the sought feature tensors.
    usr_x, itm_pos_x, itm_neg_x = utils.get_triplet_xs(data, x,
        src_node=usr_node,
        dst_node=itm_node
    )
    # Computes the link scores.
    pos_edge_score = module.regressor(usr_x, itm_pos_x)
    neg_edge_score = module.regressor(usr_x, itm_neg_x)
    # Isolates the node embeddings.
    node_embs = utils.get_embeddings(
        embedding=module.embedding, 
        data=data,
        src_node=usr_node,
        dst_node=itm_node
    )
    # Computes the loss.
    loss = loss_fn(pos_edge_score, neg_edge_score, 
        params=node_embs
    )
    # Returns the loss.
    return loss


def eval_binary(
    module: MF,
    data: HeteroData, 
    loss_fn: callable,
    *,
    edge_type: EdgeType,
    device: device
) -> Tensor:
    # Unpacks the given edge_type.
    usr_node, _, itm_node = edge_type
    # Sends the items to the correct device.
    data = data.to(device)
    # Computes the edge scores.
    edge_score = module.forward(
        usr_n_id=(usr_node, data[usr_node].n_id),
        itm_n_id=(itm_node, data[itm_node].n_id),
        edge_label_index=data[edge_type].edge_label_index
    )
    # Computes the loss.
    loss = loss_fn(edge_score, data[edge_type].edge_label)
    # Returns the computed scores.
    return loss


@torch.no_grad()
def pred(
    module: MF,
    data: HeteroData, 
    *,
    edge_type: EdgeType,
    device: device
) -> Tensor:
    # Unpacks the given edge_type.
    usr_node, _, itm_node = edge_type
    # Sends the items to the correct device.
    data = data.to(device)
    # Extracts the edge (label) index.
    try:
        edge_label_index = data[edge_type].edge_label_index
    except AttributeError:
        edge_label_index = data[edge_type].edge_index
    # Computes the edge scores.
    scores = module.forward(
        usr_n_id=(usr_node, data[usr_node].n_id),
        itm_n_id=(itm_node, data[itm_node].n_id),
        edge_label_index=edge_label_index
    )
    # Returns the computed scores.
    return scores