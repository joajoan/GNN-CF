import torch
import torch_geometric as pyg

from torch import Tensor
from torch.nn import Module
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from torch import device
from . import utils


from .base import InnerProduct, NodeEmbedding


__all__ = (
    
    # Module.
    'LightGCN',

    # Function.
    'evaluate',
    'predict'
)


class LGConv(Module):
    
    def forward(self, 
        src_x: Tensor,
        dst_x: Tensor, 
        edge_index: Tensor,
        edge_weight: Tensor
    ) -> Tensor:
        # Unpacks the edge_index.
        src_idx, dst_idx = edge_index
        # Computes the new embedding.
        msg = edge_weight * src_x[src_idx]
        # Generates the new embeddings.
        emb = pyg.utils.scatter(msg, 
            index=dst_idx, 
            dim=0, 
            dim_size=dst_x.size(0)
        )
        # Returns the new embeddings.
        return emb
    

class LGCProp(LGConv):

    def __init__(self, 
        weights: int | list[float],
        *,
        undirected: bool = True
    ) -> None:
        super().__init__()
        if type(weights) == int:
            weights = [1 / (weights + 1)] * (weights + 1)
        self.weights = weights
        self.undirected = bool(undirected)

    
    def extra_repr(self) -> str:
        return '{weights}, undirected={undirected}'.format(
            weights='[' + ', '.join([
                '{:.2f}'.format(weight) for weight in self.weights
            ]) + ']',
            undirected=self.undirected
        )
        
    
    def forward(self, 
        usr_x: Tensor,
        itm_x: Tensor,
        usr_edge_index: Tensor,
        itm_edge_index: Tensor
    ) -> Tensor:
        # Updates the edge indices to be undirected.
        if self.undirected:
            usr_edge_index = utils.make_undirected(
                src_edge_index=usr_edge_index, 
                dst_edge_index=itm_edge_index
            )
            itm_edge_index = utils.make_undirected(
                src_edge_index=itm_edge_index, 
                dst_edge_index=usr_edge_index
            )
        # Generates the edge weights for both node types.
        usr_edge_weight = utils.make_weight(usr_edge_index)
        itm_edge_weight = utils.make_weight(itm_edge_index)
        # Constructs the new embeddings.
        new_usr_x = torch.zeros_like(usr_x)
        new_itm_x = torch.zeros_like(itm_x)
        for index, weight in enumerate(self.weights):
            if index != 0:
                tmp_usr_x = super().forward(
                    src_x=itm_x, 
                    dst_x=usr_x,                             
                    edge_index=itm_edge_index, 
                    edge_weight=itm_edge_weight
                )
                tmp_itm_x = super().forward(
                    src_x=usr_x, 
                    dst_x=itm_x,                             
                    edge_index=usr_edge_index, 
                    edge_weight=usr_edge_weight
                )
                usr_x = tmp_usr_x
                itm_x = tmp_itm_x
            new_usr_x += weight * usr_x
            new_itm_x += weight * itm_x
        # Returns the embeddings.
        return new_usr_x, new_itm_x
    

class LGCEmbedding(Module):

    def __init__(self, 
        num_embeddings: dict[NodeType, int],
        embedding_dim: int,
        weights: int | list[float],
        undirected: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.embedding = NodeEmbedding(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
            **kwargs
        )
        self.propagation = LGCProp(
            weights=weights,
            undirected=undirected
        )

    
    def forward(self, 
        usr_n_id: tuple[NodeType, Tensor], 
        itm_n_id: tuple[NodeType, Tensor],
        usr_edge_index: Tensor, 
        itm_edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        # Unpacks the input arguments.
        usr_node, usr_n_id = usr_n_id
        itm_node, itm_n_id = itm_n_id
        # Constructs the base embeddings.
        usr_x, itm_x = self.embedding({
            usr_node: usr_n_id,
            itm_node: itm_n_id
        }).values()
        # Propagates the node embeddings.
        usr_x, itm_x = self.propagation(usr_x, itm_x,
            usr_edge_index=usr_edge_index,
            itm_edge_index=itm_edge_index
        )
        # Returns the LGC embeddings.
        return usr_x, itm_x
    

class LightGCN(Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.embedding = LGCEmbedding(*args, **kwargs)
        self.regressor = InnerProduct()

    
    def forward(self, 
        usr_n_id: tuple[NodeType, Tensor], 
        itm_n_id: tuple[NodeType, Tensor],
        usr_edge_index: Tensor,
        itm_edge_index: Tensor,
        edge_label_index: Tensor, 
    ) -> Tensor:
        # Unpacks the edge label index.
        usr_idx, itm_idx = edge_label_index
        # Computes the LGC embeddings.
        usr_x, itm_x = self.embedding(
            usr_n_id=usr_n_id,
            itm_n_id=itm_n_id,  
            usr_edge_index=usr_edge_index,
            itm_edge_index=itm_edge_index
        )
        # Computes the edge scores.
        edge_score = self.regressor(
            src_x=usr_x[usr_idx], 
            dst_x=itm_x[itm_idx]
        )
        # Returns the estimated edge scores.
        return edge_score
    

def eval_triplet(
    module: LightGCN, 
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
    # Extracts the user and item edge indices.
    usr_edge_index = data[usr_node, itm_node].edge_index
    itm_edge_index = data[itm_node, usr_node].edge_index
    # Computes the embedding propegation.
    usr_x, itm_x = module.embedding(
        usr_n_id=(usr_node, data[usr_node].n_id),
        itm_n_id=(itm_node, data[itm_node].n_id),
        usr_edge_index=usr_edge_index,
        itm_edge_index=itm_edge_index
    )
    # Extracts the sought feature tensors.
    usr_x, itm_pos_x, itm_neg_x = utils.get_triplet_xs(data, 
        x={
            usr_node: usr_x, 
            itm_node: itm_x
        },
        src_node=usr_node,
        dst_node=itm_node
    )
    # Computes the link scores.
    pos_edge_score = module.regressor(usr_x, itm_pos_x)
    neg_edge_score = module.regressor(usr_x, itm_neg_x)
    # Isolates the node embeddings.
    node_embs = utils.get_embeddings(
        embedding=module.embedding.embedding, 
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
    module: LightGCN,
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
        usr_edge_index=data[usr_node, itm_node].edge_index,
        itm_edge_index=data[itm_node, usr_node].edge_index,
        edge_label_index=data[edge_type].edge_label_index
    )
    # Computes the loss.
    loss = loss_fn(edge_score, data[edge_type].edge_label)
    # Returns the computed scores.
    return loss


@torch.no_grad()
def pred(
    module: LightGCN, 
    data: HeteroData, 
    *,
    edge_type: EdgeType,
    device: device
) -> Tensor:
    # Unpacks the given edge_type.
    usr_node, _, itm_node = edge_type
    # Sends the items to the correct device.
    data = data.to(device)
    # Extracts the user and item edge indices.
    usr_edge_index = data[usr_node, itm_node].edge_index
    itm_edge_index = data[itm_node, usr_node].edge_index
    # Extracts the edge (label) index.
    try:
        edge_label_index = data[edge_type].edge_label_index
    except AttributeError:
        edge_label_index = data[edge_type].edge_index
    # Computes the edge scores.
    scores = module(
        usr_n_id=(usr_node, data[usr_node].n_id),
        itm_n_id=(itm_node, data[itm_node].n_id),
        usr_edge_index=usr_edge_index,
        itm_edge_index=itm_edge_index,
        edge_label_index=edge_label_index
    )
    # Returns the computed scores.
    return scores