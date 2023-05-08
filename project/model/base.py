import torch
from torch import Tensor
from torch.nn import Embedding, Linear, Module, ModuleDict
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch.nn.init import xavier_uniform_


__all__ = (

    # Module.
    'EdgeRegressor',
    'NodeEmbedding',
    'InnerProduct',

    # Function.
    'triplet_handler'
)


def triplet_handler(
    data: HeteroData, 
    x: dict[NodeType, Tensor],
    *,
    src_node: NodeType,
    dst_node: NodeType
) -> tuple[Tensor, Tensor, Tensor]:
    # Extracts the source and destination nodes' indices.
    src_idx = data[src_node].src_index
    dst_pos_idx = data[dst_node].dst_pos_index
    dst_neg_idx = data[dst_node].dst_neg_index
    # Constructs the nodes' feature matrices.
    src_x = x[src_node][src_idx]
    dst_pos_x = x[dst_node][dst_pos_idx]
    dst_neg_x = x[dst_node][dst_neg_idx]
    # Returns the source, positive and negative features.
    return src_x, dst_pos_x, dst_neg_x


class NodeEmbedding(ModuleDict):

    def __init__(self, 
        num_embeddings: dict[NodeType, int],
        embedding_dim: int,
        **kwargs
    ) -> None:
        super().__init__({
            node_type: Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim, 
                **kwargs
            ) 
                for node_type, num_embeddings 
                in num_embeddings.items()
        })
        for embedding in self.values():
            embedding.weight.data = xavier_uniform_(embedding.weight.data)
            


    def forward(self, n_id: dict[NodeType, Tensor]) -> dict[NodeType, Tensor]:
        return {
            node_type: self[node_type](n_id) 
                for node_type, n_id 
                in n_id.items()
        }

    
class InnerProduct(Module):

    def forward(self, x_src: Tensor, x_dst: Tensor) -> Tensor:
        return torch.bmm(
            x_src.unsqueeze(-2),
            x_dst.unsqueeze(-1)
        ).squeeze()
    

class EdgeRegressor(Linear):

    def __init__(self, 
        in_dim: int, 
        out_dim: int = 1, 
        bias: bool = False, 
        **kwargs
    ) -> None:
        super().__init__(
            in_features=in_dim,
            out_features=out_dim,
            bias=bias,
            **kwargs
        )
        self.weight.data = torch.nn.init.ones_(self.weight.data)
        if self.bias:
            self.bias.data = torch.nn.init.zeros_(self.bias.data)


    def forward(self, x_src: Tensor, x_dst: Tensor) -> Tensor:
        return super().forward(x_src * x_dst)