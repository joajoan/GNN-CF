import torch
import torch_geometric as pyg

from torch import Tensor
from torch.nn import (
    Dropout1d,
    Module, 
    ModuleDict,
    ModuleList, 
    Linear, 
    LeakyReLU
)
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from itertools import pairwise
from torch import device

from .base import InnerProduct, NodeEmbedding
from .base import triplet_handler


class EmbeddingPropagationCell(Module):
    
    def __init__(self, 
        in_dim: int,
        out_dim: int = None, 
        bias: bool = False,
        dropout: float = .5
    ) -> None:
        super().__init__()
        self.drop = Dropout1d(dropout)
        self.loop = Linear(in_dim, out_dim or in_dim, bias=bias)
        self.intr = Linear(in_dim, out_dim or in_dim, bias=bias)
        self.actv = LeakyReLU()

    
    def forward(self, 
        x_src: Tensor, 
        x_dst: Tensor, 
        edge_index: Tensor,
        edge_weight: Tensor = None
    ) -> Tensor:
        # Applies the node dropout.
        x_src = self.drop(x_src)  # node dropout
        x_dst = self.drop(x_dst)  # node dropout
        # Computes the messages to pass.
        i_src, i_dst = edge_index
        z_src = self.loop(x_src)[i_src]
        z_int = self.intr(x_src[i_src] * x_dst[i_dst])
        z_msg = edge_weight * (z_src + z_int)
        z_msg = self.drop(z_msg)  # message dropout
        z_sum = pyg.utils.scatter(z_msg, i_dst, 
            dim_size=x_dst.size(0)
        )
        # Computes the self-messages.
        z_dst = self.loop(x_dst)
        z_dst = self.drop(z_dst)  # message dropout
        # Computes the new embeddings and returns them.
        x_new = self.actv(z_dst + z_sum)
        return x_new
    

class EmbeddingPropagationLayer(ModuleDict):

    def __init__(self, 
            edge_types: list[EdgeType], 
            in_dim: int, 
            out_dim: int = None, 
            **kwargs
        ) -> None:
        super().__init__({
            edge_label: EmbeddingPropagationCell(
                in_dim=in_dim, 
                out_dim=out_dim, 
                **kwargs
            )
                for (_, edge_label, _)
                in edge_types
        })


    def forward(self, 
        x: dict[NodeType, Tensor], 
        edge_index: dict[EdgeType, Tensor], 
        edge_weight: dict[EdgeType, Tensor]
    ) -> dict[NodeType, Tensor]:
        return {
            dst_node: self[edge_label](
                x_src=x[src_node], 
                x_dst=x[dst_node], 
                edge_index=edge_index[
                    src_node, edge_label, dst_node
                ],
                edge_weight=edge_weight[
                    src_node, edge_label, dst_node
                ]
            )
                for src_node, edge_label, dst_node 
                in edge_index
        }
    

class EmbeddingPropagation(ModuleList):
    
    def __init__(self, 
        embedding_dims: list[int], 
        edge_types: list[EdgeType],
        **kwargs
    ) -> None:
        super().__init__([
            EmbeddingPropagationLayer(
                edge_types=edge_types,
                in_dim=in_dim,
                out_dim=out_dim,
                **kwargs
            ) 
                for in_dim, out_dim 
                in pairwise(embedding_dims)
        ])


    def forward(self, 
        x: dict[NodeType, Tensor], 
        edge_index: dict[EdgeType, Tensor]
    ) -> dict[NodeType, Tensor]:
        # Constructs the edge weights.
        edge_weight = {
            edge_type: (
                pyg.utils.degree(i_src)[i_src]
                *
                pyg.utils.degree(i_dst)[i_dst]
            ).pow(-.5).unsqueeze(-1)
            for edge_type, (i_src, i_dst)
            in edge_index.items()
        }
        # Applies the embedding propagation layers.
        xs = [x]
        for module in self:
            x = module(x, edge_index, edge_weight)
            xs.append(x)
        # Concatenates all layers' embeddings and returns them.
        x_new = {
            node_type: torch.cat([
                    x_[node_type] for x_ in xs
            ], dim=-1) for node_type in x.keys()
        }
        return x_new

    
class NGCF(Module): 

    def __init__(self, 
        num_embeddings: dict[NodeType, int], 
        embedding_dims: list[int], 
        *,
        edge_types: list[EdgeType],
        src_node: NodeType,
        dst_node: NodeType,
        **kwargs
    ) -> None:
        super().__init__()
        self.embedding = NodeEmbedding(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dims[0]
        )
        self.propagation = EmbeddingPropagation(
            embedding_dims=embedding_dims, 
            edge_types=edge_types,
            **kwargs
        )
        self.regressor = InnerProduct()
    

    def forward(self, 
            n_id: dict[NodeType, Tensor],
            edge_index: dict[EdgeType, Tensor],
            edge_label_index: Tensor
        ) -> Tensor:
        # Generates and propagates the node embeddings.
        x = self.embedding(n_id)
        x = self.propagation(x, edge_index)
        # Computes the rank predictions and returns them.
        y = self.regressor(
            x[self.src_node][edge_label_index[0]], 
            x[self.dst_node][edge_label_index[1]]
        )
        # Returns the modified dataset.
        return y
    


def evaluate(
    module: NGCF, 
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
    module: NGCF, 
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


def evaluate(
    module: type[Module], 
    data: HeteroData, 
    criterion: callable,
    *,
    device: torch.device
) -> Tensor:
    
    # Sends the items to the correct device.
    data = data.to(device)

    # Computes the embedding propegation.
    x = module.embedding(data.n_id_dict)
    x = module.propagation(x, data.edge_index_dict)
    # Extracts the source nodes' features.
    i_src = data['user'].src_index
    x_src = x['user'][i_src]
    # Constructs the positive and negative feature matrices.
    i_pos = data['item'].dst_pos_index
    i_neg = data['item'].dst_neg_index
    x_pos = x['item'][i_pos]
    x_neg = x['item'][i_neg]
    # Computes the link scores.
    y_pos = module.regressor(x_src, x_pos)
    y_neg = module.regressor(x_src, x_neg)

    # Computes the loss.
    loss = criterion(y_pos, y_neg)

    # Returns the loss.
    return loss