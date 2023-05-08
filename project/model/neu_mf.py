class GMF(EdgePredictor):

    def __init__(self, 
            num_embeddings: dict[NodeType, int], 
            embedding_dim: int,
            *,
            trg_edge: EdgeType,
            **kwargs
        ) -> None:
        super().__init__(trg_edge=trg_edge)
        self.embedding = NodeEmbedding(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
            **kwargs
        )
        self.regressor = EdgeRegressor(
            in_dim=embedding_dim
        )
        

    def forward(self, 
        n_id: dict[NodeType, Tensor], 
        edge_label_index: dict[EdgeType, Tensor], 
    ) -> Tensor:
        # Constructs the embeddings.
        x_src, x_dst = self.embedding({
            node_type: n_id[node_type] 
            for node_type 
            in self.trg_nodes
        }).values()
        # Extracts the edges to predict.
        i_src, i_dst = edge_label_index[self.trg_edge]
        # Computes and returns the predicted scores.
        return self.regressor(x_src[i_src], x_dst[i_dst])