import torch

from torch import Tensor
from torch.nn import Module
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from torch import device

from .base import InnerProduct, NodeEmbedding
from .utils import triplet_handler


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
    

    def evaluate(self,
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
        x = self.embedding({
            usr_node: data[usr_node].n_id,
            itm_node: data[itm_node].n_id
        })
        # Extracts the sought feature tensors.
        usr_x, itm_pos_x, itm_neg_x = triplet_handler(data, x,
            src_node=usr_node,
            dst_node=itm_node
        )
        # Computes the link scores.
        pos_edge_score = self.regressor(usr_x, itm_pos_x)
        neg_edge_score = self.regressor(usr_x, itm_neg_x)
        # Computes the loss.
        loss = loss_fn(pos_edge_score, neg_edge_score)
        # Returns the loss.
        return loss


    @torch.no_grad()
    def predict(self,
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
        scores = self.forward(
            usr_n_id=(usr_node, data[usr_node].n_id),
            itm_n_id=(itm_node, data[itm_node].n_id),
            edge_label_index=edge_label_index
        )
        # Returns the computed scores.
        return scores