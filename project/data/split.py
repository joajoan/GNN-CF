# Namespace.
import torch
import os

# Capital.
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import HeteroData
from argparse import ArgumentParser


def save(data: HeteroData, path: str, name: str) -> None:
    # Constructs the output file-path.
    path = os.path.join(path, name)
    # Saves the given dataset.
    torch.save(data, path)


def run(
    source: str, 
    target: str,
    *,
    num_test: float,
    num_val: float,
    seed: int = None,
    keep_attr: bool = False
) -> None:

    # Validates the source file-path.
    assert os.path.isfile(source)
    # Constructs target path.
    if target is None:
        target = os.path.dirname(source)
    assert os.path.isdir(target)
    # Loads the heterogeneous dataset.
    data = torch.load(source)

    # Removes the attributes, if not specified otherwize.
    if not keep_attr:
        for edge_type in data.edge_types:
            if 'edge_attr' in data[edge_type]:
                del data[edge_type].edge_attr
    # Setting the random seed, if specified.
    if seed is not None:
        torch.manual_seed(seed)
    # Defines the final graph transformations.
    transform = RandomLinkSplit(
        num_test=num_test,
        num_val=num_val,
        is_undirected=True, 
        add_negative_train_samples=False,
        neg_sampling_ratio=0.,
        edge_types=('user', 'rated', 'item'),
        rev_edge_types=('item', 'rated_by', 'user')
    )
    # Splits the set into training, validation and testing.
    trn_data, vld_data, tst_data = transform(data)

    # Infers the source's file-name.
    filename = os.path.basename(source)
    # Saves the partitioned dataset.
    save(trn_data, target, 'trn_' + filename)
    save(vld_data, target, 'vld_' + filename)
    save(tst_data, target, 'tst_' + filename)


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
    parser.add_argument('--num_test', 
        type=float, 
        help='The sought fraction or absolute number of test edges', 
        metavar='NUM',
        default=1e-3,
    )
    parser.add_argument('--num_val', 
        type=float, 
        help='The sought fraction or absolute number of validation edges', 
        metavar='NUM',
        default=1e-3,
    )
    parser.add_argument('--seed', 
        type=int, 
        help='Random seed for when partitioning the dataset', 
        metavar='SEED',
        default=None,
    )
    parser.add_argument('--keep_attr', 
        action='store_true',
        help='Specifices that the edge attributes should be kept', 
    )
    args = parser.parse_args(*args)

    # Runs the program.
    run(
        source=args.source, 
        target=args.target, 
        num_test=args.num_test,
        num_val=args.num_val,
        seed=args.seed,
        keep_attr=args.keep_attr
    )


if __name__ == '__main__':
    main()