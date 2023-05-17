import torch
from torch import Tensor
from torch.nn import Embedding, Module, ModuleDict
from torch_geometric.typing import NodeType
from torch.nn import init


__all__ = (
    'NodeEmbedding',
    'InnerProduct'
)


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
            embedding.weight.data = init.xavier_uniform_(embedding.weight.data)
            


    def forward(self, 
        n_id: dict[NodeType, Tensor] | list[Tensor]
    ) -> dict[NodeType, Tensor]:
        return {
            node_type: self[node_type](n_id) 
                for node_type, n_id 
                in n_id.items()
        } if type(n_id) == dict else [
            module(n_id) 
                for n_id, module 
                in zip(n_id, self.values())
        ]

    
class InnerProduct(Module):

    def forward(self, src_x: Tensor, dst_x: Tensor) -> Tensor:
        return torch.bmm(
            src_x.unsqueeze(-2),
            dst_x.unsqueeze(-1)
        ).squeeze()