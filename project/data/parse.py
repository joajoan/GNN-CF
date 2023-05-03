# Namespace.
import torch
import json
import os

# Capital.
from torch_geometric.data import HeteroData
from argparse import ArgumentParser


def load_data(path: str) -> list[dict]:
    buffer = []
    with open(path) as file:
        for item in file:
            item = json.loads(item)
            buffer.append([
                item['reviewerID'],  # user ID
                item['asin'],  # item ID
                item['overall']  # rating
            ])
    return buffer


def make_data(data: list[tuple]) -> HeteroData:

    # Extracts the user, item and review values.
    user_list, item_list, rate_list = zip(*data)
    # Infers the unique user and item labels.
    unique_users = sorted(set(user_list))
    unique_items = sorted(set(item_list))
    # Creates mappings from labels to index positions.
    user_map = {user: index for index, user in enumerate(unique_users)}
    item_map = {item: index for index, item in enumerate(unique_items)}

    # Creates the adjecency list from the ratings.
    rate_edge = torch.tensor([
        (user_map[user], item_map[item]) 
            for user, item 
            in zip(user_list, item_list)
    ]).transpose(0, 1)
    # Reformats the score ratings, for both edge types.
    edge_attr = torch.tensor(rate_list).unsqueeze(-1)

    # Creates a heterogeneous graph data object.
    data = HeteroData()
    # Defines the number of the user and item nodes.
    data['user'].num_nodes = len(unique_users)
    data['item'].num_nodes = len(unique_items)
    # Specifies the edges in the graph, including the reverse links.
    data['user', 'rated', 'item'].edge_index = rate_edge
    data['item', 'rated_by', 'user'].edge_index = rate_edge.flip(0)
    # Specifies the weights of the edges.
    data['user', 'rated', 'item'].edge_attr = edge_attr
    data['item', 'rated_by', 'user'].edge_attr = edge_attr
    
    # Sets the node and edge IDs.
    data.generate_ids()

    # Returns the graph dataset.
    return data


def run(source: str, target: str) -> None:

    # Ensures a valid source file-path.
    assert os.path.isfile(source)
    # Ensures a valid target file-path.
    if target is None:
        target = os.path.dirname(source)
    assert os.path.exists(target)
    if os.path.isdir(target):
        # Infers the name stem of the source file.
        name = os.path.basename(source)
        stem = os.path.splitext(name)[0]
        # Creates the target file-path.
        target = os.path.join(target, stem + '.pt')

    # Loads and parses the file content into memory.
    data = load_data(source)
    # Creates a heterogeneous dataset from the parsed file contents.
    data = make_data(data)
    # Saves the dataset to disk.
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
        default=None
    )
    args = parser.parse_args(*args)

    # Runs the program.
    run(source=args.source, target=args.target)


if __name__ == '__main__':
    main()