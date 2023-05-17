import torch
import tqdm
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch_geometric.typing import EdgeType
from . import get_device


__all__ = ('rank_k',) 


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
) -> tuple[Tensor, Tensor]:
    
    # Resolves the device.
    device = get_device(device)
    # Empties the GPU cache, if that device is set.
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Unpacks the given edge type.
    src_node, _, _ = edge_type

    # Identifies the total number of source node and their respective IDs.
    gbl_uid = loader.data[edge_type].edge_label_index[0].unique().to(device)
    gbl_cnt = gbl_uid.numel()
    # Setting up the storage buffers.
    scr_buf = torch.full([gbl_cnt, 2 * at_k], 
        fill_value=-torch.inf, 
        device=device
    )  # predicted scores
    lbl_buf = torch.empty(gbl_cnt, 2 * at_k, device=device)  # ground-truth labels
    pos_cnt = torch.zeros(gbl_cnt, dtype=torch.int, device=device)  # relevant item counts

    # Wraps the loader in a progress tracker, if verbose is set.
    if verbose is True:
        loader = tqdm.tqdm(loader, mininterval=1., position=0, leave=True)
    # Iterates over batched edges from the loader.
    for batch in loader:

        # Predicts the edge-wise score.
        edge_score = pred_fn(module, batch, 
            edge_type=edge_type, 
            device=device
        )
        # Extracts the edge label tensor.
        edge_label = batch[edge_type].edge_label
        
        # Unpacks the edge label indices.
        src_idx, _ = batch[edge_type].edge_label_index
        # Identifies what node IDs the indices represent.
        src_ids = batch[src_node].n_id[src_idx]
        # Infers the relevant source node IDs and their counts.
        src_uid, uid_cnt = src_ids.unique(return_counts=True)

        # Infers the source nodes in the tracked set.
        gbl_msk = (
            gbl_uid.unsqueeze(-1) 
                == 
            src_uid.unsqueeze(-2)
        ).any(dim=-1)

        # Infers the temporary buffer sizes.
        row_cnt = gbl_msk.sum().item()
        col_cnt = max(at_k, uid_cnt.max().item())
        # Constructs the temporary buffer sizes.
        scr_tmp = torch.full([row_cnt, col_cnt], 
            fill_value=-torch.inf, 
            device=device
        )
        lbl_tmp = torch.empty(row_cnt, col_cnt, device=device)

        # Builds the second dimension's indices.
        col_idx = torch.cat([
            torch.arange(cnt, device=device) for cnt in uid_cnt
        ])
        # Fills the buffers.
        scr_tmp[src_idx, col_idx] = edge_score
        lbl_tmp[src_idx, col_idx] = edge_label

        # Selects the top-k elements.
        scr_tmp, top_idx = torch.topk(scr_tmp, k=at_k, dim=-1, sorted=False)
        lbl_tmp = lbl_tmp.gather(dim=1, index=top_idx)

        # Updates buffers.
        scr_buf[gbl_msk, at_k:] = scr_tmp
        lbl_buf[gbl_msk, at_k:] = lbl_tmp
        # Sorts the main buffers.
        buf_idx = scr_buf.argsort(dim=-1, descending=True)
        # Updates the buffer orderings.
        scr_buf = scr_buf.gather(dim=-1, index=buf_idx)
        lbl_buf = lbl_buf.gather(dim=-1, index=buf_idx)

        # Updates the total positive edges per source node.
        pos_cnt[gbl_msk] += torch.zeros(row_cnt, 
            dtype=torch.int,
            device=device
        ).scatter_add(
            dim=0, 
            index=src_idx, 
            src=edge_label.int()
        )

    # Returns the top-k ranked labels and the total positive instances.
    return lbl_buf.cpu(), pos_cnt.cpu()