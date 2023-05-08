import torch
import tqdm
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch_geometric.typing import EdgeType


__all__ = (
    'ndcg_coefs',
    'ndcg_score',
    'recall_score',
    'composite'
)


def recall_score(score: Tensor, total: Tensor) -> Tensor:
    '''Computes recal'''
    return score.sum(dim=-1) / total


def ndcg_coefs(count: int) -> Tensor:
    '''Creats the discount factors used in normalized discounted cumulative gain.'''
    coefs = torch.arange(count) + 2
    coefs = torch.log2(coefs)
    return coefs


def ndcg_score(
    score: Tensor, 
    total: Tensor,
    *, 
    coefs: Tensor = None
) -> Tensor:
    '''...'''
    
    # Infers the score's resolution.
    *_, batch, count = score.shape
    # Converts the total to a tensor, if given an integer.
    if type(total) == int:
        total = torch.full([count], count)
    # Constructs the discount factors, if not given.
    if coefs is None:
        coefs = ndcg_coefs(count)

    # Determines terms for the normalization summation.
    tally = torch.full_like(total, count)
    terms = (
        torch.arange(count).unsqueeze(-2).expand(batch, count)
        <
        torch.stack([total, tally]).amin(-2).long().unsqueeze(-1)
    ) * coefs.unsqueeze(-2).expand(batch, count)
    
    # Computes and returns the NDCG-score.
    return (score / coefs).sum(-1) / terms.sum(-1)


def rank(
    module: type[Module],
    loader: type[DataLoader],
    *,
    pred_fn: callable,
    edge_type: EdgeType,
    device: torch.device,
    at_k: int = 20,
    verbose: bool = False
) -> tuple[Tensor, Tensor]:
    
    # Empties the GPU cache, if that device is set.
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Unpacks the given edge type.
    src_node, _, _ = edge_type

    # Wraps the loader in a progress measurer, if verbose is set.
    if verbose is True:
        loader = tqdm.tqdm(loader, mininterval=1., position=0, leave=True)
    # Creates the hashmap that is to be housing the output buffers.
    buffer = {}
    # Iterates over the data-loader's batches.
    for batch in loader:

        # Predicts edge-wise scores.
        edge_score = pred_fn(module,
            data=batch,
            edge_type=edge_type,
            device=device
        )

        # Unpacks the edge label index.
        src_idx = batch[edge_type].edge_label_index[0]
        # Derives the source and destination node IDs.
        src_id = batch[src_node].n_id[src_idx]
        # Extracts the edge labels.
        edge_label = batch[edge_type].edge_label

        # Identifies the unique source node IDs in the batch.
        src_uid = src_id.unique()
        # Computes the source node ID boolean map.
        src_mask = src_uid.unsqueeze(1) == src_id.unsqueeze(0)
        
        # Updates the relevant source nodes' buffers.
        for uid, mask in zip(src_uid, src_mask):

            # Parses the unique ID of the source node.
            uid = uid.item()
            # Ensures the buffers exists.
            if uid not in buffer:
                buffer[uid] = ([], [])

            # Updates the given source node's buffers.
            scr_lst, lbl_lst = buffer[uid]
            scr_lst.append(edge_score[mask].cpu())
            lbl_lst.append(edge_label[mask].cpu())

    # Concatenates the inner buffers to pure tensors.
    buffer = [
        tuple(
            torch.cat(buffer) for buffer in buffers
        ) 
        for buffers in buffer.values()
    ]

    # Computes the top-k item labels.
    lbl_top = []
    pos_cnt = []
    for scr_buf, lbl_buf in buffer:
        top_idx = torch.topk(scr_buf, k=at_k).indices
        lbl_top.append(lbl_buf[top_idx])
        pos_cnt.append(lbl_buf.sum())
    lbl_top = torch.stack(lbl_top)
    pos_cnt = torch.tensor(pos_cnt)

    # Returns the top-k labels and the total positive edge count.
    return lbl_top, pos_cnt


def composite(
    *args,
    score_fns: list[callable],
    verbose: bool = False,
    **kwargs
) -> list[float]:
    
    # Computes the ranked labels and the total possitive instances.
    labels, counts = rank(*args, verbose=verbose, **kwargs)

    # Computes the scores 
    scores = [score_fn(labels, counts).mean() for score_fn in score_fns]

    # Returns the scores as a tensor.
    return torch.tensor(scores).tolist()