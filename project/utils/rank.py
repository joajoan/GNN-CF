import torch
import tqdm
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch_geometric.typing import EdgeType
from typing import NamedTuple
from . import get_device


__all__ = ('rank_k',) 


class Ranking(NamedTuple):
    score: Tensor
    label: Tensor
    count: Tensor
    item: Tensor
    user: Tensor


@torch.no_grad()
def rank_k(
    module: type[Module],
    loader: type[DataLoader],
    *,
    pred_fn: callable,
    at_k: int = 20,
    edge_type: EdgeType,
    device: torch.device = None,
    verbose: bool = False
) -> Ranking:
    
    # Resolves the device.
    device = get_device(device)
    # Empties the GPU cache, if that device is set.
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Setting up the model for evaluation.
    module = module.to(device).eval()
    # Unpacks the given edge type.
    src_node, _, dst_node = edge_type

    # Identifies the total number of source node and their respective IDs.
    gbl_idx = loader.data[edge_type].edge_label_index[0].unique()
    gbl_uid = loader.data[src_node].n_id[gbl_idx].to(device)

    # Wraps the loader in a progress tracker object, if verbose is set.
    if verbose is True:
        loader_ = tqdm.tqdm(loader, mininterval=1., position=0, leave=True)
    else:
        loader_ = loader
    # Creates the hash-map that is to be housing the output buffers.
    score_buffer = [list() for _ in gbl_uid]
    label_buffer = [list() for _ in gbl_uid]
    id_buffer = [list() for _ in gbl_uid]
    # Iterates over the data-loader's batches.
    for batch in loader_:

        # Predicts edge-wise scores.
        edge_score = pred_fn(module,
            data=batch,
            edge_type=edge_type,
            device=device
        )

        # Unpacks the edge label index.
        src_idx, dst_idx = batch[edge_type].edge_label_index
        # Derives the source and destination node IDs.
        src_id = batch[src_node].n_id[src_idx]
        dst_id = batch[dst_node].n_id[dst_idx]
        # Extracts the edge labels and destination node IDs.
        edge_label = batch[edge_type].edge_label

        # Identifies the unique source node IDs in the batch.
        src_uid = src_id.unique()
        # Infers the relevant source nodes in the tracked set.
        out_idx = (
            gbl_uid.unsqueeze(1) 
                == 
            src_uid.unsqueeze(0)
        ).any(dim=-1).nonzero().ravel()
        # Generates the relevant nodes' element mask.
        out_msk = gbl_uid[out_idx].unsqueeze(1) == src_id.unsqueeze(0)
        
        # Updates the relevant source nodes' buffers.
        for idx, msk in zip(out_idx, out_msk):
            score_buffer[idx].append(edge_score[msk].cpu())
            label_buffer[idx].append(edge_label[msk].cpu())
            id_buffer[idx].append(dst_id[msk].cpu())

    # Concatenates all sub-buffers for the score, label and ID buffers.
    scores = [torch.cat(buffer) for buffer in score_buffer]
    labels = [torch.cat(buffer) for buffer in label_buffer]
    ids = [torch.cat(buffer) for buffer in id_buffer]

    # Initializes the positive edge counter.
    positive_count = []
    # Initializes the top-k score, label and id buffers.
    score_top, label_top, id_top = [], [], []
    # Computes the top-k item scores, labels and ids.
    for scores, labels, ids in zip(scores, labels, ids):

        # Computes the top-k values and their indices.
        scores, index = torch.topk(scores, k=at_k)

        # Saves the top-k scores, labels and ids.
        score_top.append(scores)
        label_top.append(labels[index])
        id_top.append(ids[index])

        # Counts the total number of positive edges.
        positive_count.append(labels.sum())

    # Reformats the top-k buffers to pure tensors.
    score = torch.stack(score_top)
    label = torch.stack(label_top)
    id = torch.stack(id_top)
    count = torch.tensor(positive_count)

    # Returns the top-k labels and the total positive edge count.
    return Ranking(score, label, count, item=id, user=gbl_uid)