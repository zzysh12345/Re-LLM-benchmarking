import os
import sys
os.chdir('../')
print(os.getcwd())
project_root = os.getcwd()
sys.path.append(project_root)
from utils.utils import load_data, get_and_save_message_for_node
import numpy as np
import torch
import pickle
import random


def create_hterophily_graph(dataset_name, drop_rate):


    name_sign = int(drop_rate*100)
    data_path = f"dataset/{name_sign}_drop/{dataset_name}_data.pt"

    if os.path.exists(data_path):
        print(f"File already exists: {data_path}")
        return

    directory = os.path.dirname(data_path)

    if not os.path.exists(directory):
        os.makedirs(directory)


    original_data_path = f"dataset/{dataset_name}_data.pt"
    if os.path.exists(original_data_path):
        data = torch.load(original_data_path)

    edge_index = data.edge_index
    edge_tuples = {(src.item(), dst.item()) if src.item() <= dst.item() else (dst.item(), src.item()) for src, dst in edge_index.t()}  # 转为集合去重
    directed_edges = torch.tensor(list(edge_tuples), dtype=torch.long)
    # data.edge_index = directed_edges

    num_classes = data.y.unique().size(0)
    num_node = data.y.size(0)
    # edge_index = data.edge_index
    # edge_index_T = edge_index.T
    # num_edge = edge_index.size(1)

    same_class_edge_count = 0
    same_class_edge = []
    different_class_edge = []
    for edge in directed_edges:
        if data.y[edge[0].item()] == data.y[edge[1].item()]:
            same_class_edge_count += 1
            same_class_edge.append(edge)
        else:
            different_class_edge.append(edge)

    sample_edge_count = int(same_class_edge_count * (1-drop_rate))
    keep_edge = random.sample(same_class_edge, sample_edge_count)
    # final_edge_index = [item for item in edge_tuples if item not in remove_edge]
    # final_edge_index = []
    # from tqdm import tqdm
    # for item in tqdm(edge_tuples, desc="Processing edges", unit="edge"):
    #     if item not in remove_edge:
    #         final_edge_index.append(item)
    final_edge_index = keep_edge + different_class_edge
    random.shuffle(final_edge_index)
    result = torch.stack(final_edge_index).t()

    from torch_geometric.utils import to_undirected
    undirected_edge_index = to_undirected(result)

    data.edge_index = undirected_edge_index


    torch.save(data, data_path)

def create_drop_edge_graph(dataset_name, drop_rate):
    name_sign = int(drop_rate * 100)
    final_save_path = f"dataset/random_{name_sign}_drop/{dataset_name}_data.pt"
    data_path = f"dataset/{name_sign}_drop/{dataset_name}_data.pt"
    if os.path.exists(data_path):
        data = torch.load(data_path)
    else:
        print(f"File does not exist: {data_path}")
        return
    edge_num = int(data.edge_index.size(1) / 2)

    original_data_path = f"dataset/{dataset_name}_data.pt"
    if os.path.exists(original_data_path):
        data = torch.load(original_data_path)
    else:
        print(f"File does not exist: {original_data_path}")
        return
    edge_index = data.edge_index
    edge_tuples = {(src.item(), dst.item()) if src.item() <= dst.item() else (dst.item(), src.item()) for src, dst in
                   edge_index.t()}
    directed_edges = torch.tensor(list(edge_tuples), dtype=torch.long)
    keep_edge = random.sample(directed_edges.tolist(), edge_num)
    keep_edge = torch.tensor(keep_edge, dtype=torch.long).t()
    from torch_geometric.utils import to_undirected
    undirected_edge_index = to_undirected(keep_edge)

    data.edge_index = undirected_edge_index

    directory = os.path.dirname(final_save_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(data, final_save_path)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process dataset_name and drop_rate")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--drop_rate', type=float, required=True, help="Drop rate for edge removal")

    args = parser.parse_args()

    dataset_name = args.dataset_name
    drop_rate = args.drop_rate

    create_hterophily_graph(dataset_name, drop_rate)
    name_sign = int(drop_rate * 100)
    data_path = f"dataset/{name_sign}_drop/{dataset_name}_data.pt"
    if os.path.exists(data_path):
        data = torch.load(data_path)

    print(data)

    create_drop_edge_graph(dataset_name, drop_rate)
    data_path = f"dataset/random_{name_sign}_drop/{dataset_name}_data.pt"
    if os.path.exists(data_path):
        data = torch.load(data_path)
    print(data)




