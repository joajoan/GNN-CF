# Namespace.
import torch
import tqdm
import os

# Captial.
from torch_geometric.typing import EdgeType
from torch_geometric.data import HeteroData
from argparse import ArgumentParser
from torch import Tensor


def make_map(edge_index: Tensor) -> dict[int, set[int]]:
    '''Creates a mapping from every head ID to its set of tail IDs.'''
    map = {}
    for src_id, dst_id in zip(*edge_index):
        src_id = src_id.item()
        dst_id = dst_id.item()
        if src_id not in map:
            map[src_id] = set()
        map[src_id].add(dst_id)
    return map


def make_all_ranking(
    data: HeteroData, 
    *, 
    trg_edge: EdgeType, 
    inplace: bool = False,
    verbose: bool = False
) -> HeteroData:
    '''Converts a heterogeneous dataset to an instance with negative edges.'''
    # Makes a copy of input data.
    if not inplace:
        data = data.clone()
    # Derives the source nodes of interest.
    edge_label_index = data[trg_edge].edge_label_index
    src_map = make_map(edge_label_index)
    # Constructs the source-to-destination map for nodes to be excluded.
    edge_index = data[trg_edge].edge_index
    exc_map = make_map(edge_index)
    exc_map = {
        src_id: (
            exc_map[src_id] | src_map[src_id] 
                if src_id in exc_map 
                else src_map[src_id]
        ) for src_id in src_map
    }
    # Initializes the new edge index tensor for the labels.
    _, _, dst_node = trg_edge
    num_dst_nodes = data[dst_node].num_nodes
    num_pos_edges = edge_label_index.size(1)
    num_neg_edges = len(src_map) * num_dst_nodes - sum([
        len(dst_ids) for dst_ids in exc_map.values()
    ])
    num_edges = num_pos_edges + num_neg_edges
    # Fills the new edge index with the positive edges.
    src_ids, dst_ids = edge_label_index
    edge_label_index = torch.empty(2, num_edges, dtype=torch.long)
    edge_label_index[0, :num_pos_edges] = src_ids
    edge_label_index[1, :num_pos_edges] = dst_ids
    _index = num_pos_edges
    # Wraps the iterable in a tqdm-object, if verbose is set.
    src_uids = list(src_map)
    if verbose is True:
        src_uids = tqdm.tqdm(src_uids)
    # Fills the new edge index with the negative edges.
    for src_id in src_uids:
        # Constructs the valid destination nodes through negative links
        dst_ids = torch.tensor([
            dst_id for dst_id in range(num_dst_nodes) 
                if dst_id not in exc_map[src_id]
        ])
        # Adds the current source and its directed edges.
        edge_label_index[0, _index:_index+len(dst_ids)] = src_id
        edge_label_index[1, _index:_index+len(dst_ids)] = dst_ids
        # Updates the starting index.
        _index += len(dst_ids)
    # Creates the new edge labels.
    edge_label = torch.zeros(num_edges, dtype=torch.float)
    edge_label[:num_pos_edges] = torch.ones(num_pos_edges)
    # Updates the data copy.
    data[trg_edge].edge_label_index = edge_label_index
    data[trg_edge].edge_label = edge_label
    # Returns a modified copy of the original data.
    return data


def run(
    source: str, 
    target: str = None,
    *,
    verbose: bool = None
) -> None:

    # Validates the source file-path.
    assert os.path.isfile(source)
    # Constructs target path.
    if target is None:
        target = os.path.dirname(source)
    assert os.path.exists(target)
    if os.path.isdir(target):
        filename = os.path.basename(source)
        target = os.path.join(target, 'rnk_' + filename)

    # Loads the heterogeneous dataset.
    data = torch.load(source)

    # Converts the dataset into an instance for the all-ranking protocol.
    data = make_all_ranking(data, 
        trg_edge=('user', 'rated', 'item'),
        inplace=True,
        verbose=verbose
    )

    # Saves the modified dataset.
    torch.save(data, target)
    

def main(*args) -> None:

    # Creates a command line argument parser.
    parser = ArgumentParser(
        description='Creates a heterogeneous graph dataset given a file from '
            'the Amazon Review project'
    )
    parser.add_argument('-s', '--source', 
        type=str, 
        help='Path to the file to create a dataset from', 
        metavar='PATH'
    )
    parser.add_argument('-t', '--target', 
        type=str, 
        help='Path to a file to store the created dataset', 
        metavar='PATH',
        default=None,
    )
    parser.add_argument('-v', '--verbose', 
        action='store_true',
        help='Specifies whether or not to be verbose about its progress', 
    )
    args = parser.parse_args(*args)

    # Runs the program.
    run(
        source=args.source, 
        target=args.target, 
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()