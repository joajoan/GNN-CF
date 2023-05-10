import torch
import torch_geometric as pyg

from torch import Tensor
from torch.nn import (
    Module, 
    ModuleList, 
    Linear, 
    LeakyReLU
)
from torch.nn.functional import dropout1d as dropout
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from itertools import pairwise
from torch.nn import init
from torch import device

from .base import InnerProduct, NodeEmbedding
from .utils import make_weight, make_undirected, triplet_handler


class EmbPropCell(Module):
    
    def __init__(self, 
        in_dim: int,
        out_dim: int = None, 
        bias: bool = False,
        node_dropout: float = 0.,
        message_dropout: float = .1
    ) -> None:
        super().__init__()
        # Saves the drouput arguments.
        self.node_dropout = node_dropout or 0.
        self.message_dropout = message_dropout or 0.
        # Initializes the interanl modules.
        self.intra_linear = Linear(in_dim, out_dim or in_dim, bias=bias)
        self.inter_linear = Linear(in_dim, out_dim or in_dim, bias=bias)
        self.activation = LeakyReLU()
        # Initializes weights.
        self.intra_linear.weight.data = init.xavier_uniform_(
            self.intra_linear.weight.data
        )
        self.inter_linear.weight.data = init.xavier_uniform_(
            self.inter_linear.weight.data
        )

    
    def forward(self, 
        src_x: Tensor, 
        dst_x: Tensor, 
        edge_index: Tensor,
        edge_weight: Tensor
    ) -> Tensor:
        # Unpacks the edge index.
        src_idx, dst_idx = edge_index
        # Applies source node dropout.
        src_x = dropout(src_x, 
            p=self.node_dropout, 
            training=self.training
        )
        # Computes the source messages.
        s2s_msgs = self.intra_linear(src_x)[src_idx]
        i2s_msgs = self.inter_linear(src_x[src_idx] * dst_x[dst_idx])
        src_msgs = edge_weight * (s2s_msgs + i2s_msgs)
        # Generates the self messages.
        self_msg = self.intra_linear(dst_x)
        # Applies message dropout.
        src_msgs = dropout(src_msgs, 
            p=self.message_dropout,
            training=self.training
        )
        self_msg = dropout(self_msg, 
            p=self.message_dropout,
            training=self.training
        )
        # Aggregates the messages.
        src_msg = pyg.utils.scatter(src_msgs, 
            index=dst_idx, 
            dim=0, 
            dim_size=dst_x.size(0)
        )
        dst_msg = self_msg + src_msg
        # Applies the activation function.
        dst_x = self.activation(dst_msg)
        # Returns the new destination features.
        return dst_x
    

class EmbPropLayer(Module):

    def __init__(self, 
            in_dim: int, 
            out_dim: int = None, 
            **kwargs
        ) -> None:
        super().__init__()
        self.user = EmbPropCell(in_dim, out_dim or in_dim, **kwargs)
        self.item = EmbPropCell(in_dim, out_dim or in_dim, **kwargs)


    def forward(self, 
        usr_x: Tensor,
        itm_x: Tensor,
        usr_edge_index: Tensor,
        itm_edge_index: Tensor,
        usr_edge_weight: Tensor,
        itm_edge_weight: Tensor
    ) -> tuple[Tensor, Tensor]:
        new_usr_x = self.user(itm_x, usr_x,
            edge_index=itm_edge_index,
            edge_weight=itm_edge_weight
        )
        new_itm_x = self.user(usr_x, itm_x,
            edge_index=usr_edge_index,
            edge_weight=usr_edge_weight
        )
        return new_usr_x, new_itm_x
    

class EmbProp(ModuleList):
    
    def __init__(self, 
        dims: int | list[int], 
        weights: list[float] = None,
        *,
        undirected: bool = True,
        **kwargs
    ) -> None:
        # Parsing the layer-wise dimensions.
        if type(dims) == int:
            dims = [dims]
        assert type(dims) == list
        if len(dims) == 1:
            dims *= 2
        assert len(dims) >= 2
        # Parses the importance weights.
        if weights is None:
            weights = [1 / len(dims)] * len(dims)
        assert len(weights) == len(dims)
        # Initailizes the internal modules.
        super().__init__([
            EmbPropLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                **kwargs
            ) 
                for in_dim, out_dim 
                in pairwise(dims)
        ])
        # Saves the configuration input arguments.
        self.weights = weights
        self.undirected = bool(undirected)

    
    def forward(self, 
        usr_x: Tensor,
        itm_x: Tensor,
        usr_edge_index: Tensor,
        itm_edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Updates the edge indices to be undirected.
        if self.undirected:
            usr_edge_index = make_undirected(usr_edge_index, itm_edge_index)
            itm_edge_index = usr_edge_index.flip(0)
        # Generates the edge weights for both node types.
        usr_edge_weight = make_weight(usr_edge_index)
        itm_edge_weight = make_weight(itm_edge_index)
        # Constructs the new embeddings.
        new_usr_x = torch.zeros_like(usr_x)
        new_itm_x = torch.zeros_like(itm_x)
        for index, weight in enumerate(self.weights, start=-1):
            if index >= 0:
                usr_x, itm_x = self[index](
                    usr_x=usr_x,
                    itm_x=itm_x,
                    usr_edge_index=usr_edge_index,
                    itm_edge_index=itm_edge_index,
                    usr_edge_weight=usr_edge_weight,
                    itm_edge_weight=itm_edge_weight,
                )
            new_usr_x += weight * usr_x
            new_itm_x += weight * itm_x
        # Returns the embeddings.
        return new_usr_x, new_itm_x
    

class NGCFEmbedding(Module):
    
    def __init__(self, 
        num_embeddings: dict[NodeType, int],
        embedding_dims: int | list[int],
        *,
        embedding_weights: list[float] = None,
        undirected: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        # Parsing the embedding layer dimensions.
        if type(embedding_dims) == int:
            embedding_dims = [embedding_dims]
        assert type(embedding_dims) == list
        if len(embedding_dims) == 1:
            embedding_dims *= 2
        assert len(embedding_dims) >= 2
        embedding_dim, *_ = embedding_dims
        # Initializes the internal modules.
        self.embedding = NodeEmbedding(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
        )
        self.propagation = EmbProp(
            dims=embedding_dims,
            weights=embedding_weights,
            undirected=undirected,
            **kwargs
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
    

class NGCF(Module): 

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.embedding = NGCFEmbedding(*args, **kwargs)
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
        # Computes edge scores.
        edge_score = self.regressor(
            src_x=usr_x[usr_idx], 
            dst_x=itm_x[itm_idx]
        )
        # Returns the estimated edge scores.
        return edge_score
    

def evaluate(
    module: NGCF, 
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
    usr_x, itm_pos_x, itm_neg_x = triplet_handler(data, 
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
    # Computes the loss.
    loss = loss_fn(pos_edge_score, neg_edge_score)
    # Returns the loss.
    return loss


@torch.no_grad()
def predict(
    module: NGCF, 
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