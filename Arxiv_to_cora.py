import os
import sys
os.chdir('../')
print(os.getcwd())
project_root = os.getcwd()
sys.path.append(project_root)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling, subgraph, to_scipy_sparse_matrix
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# from ogb.nodeproppred import PygNodePropPredDataset
import os
import pickle
from utils.utils import load_data
from model import GCN, GAT, GraphSAGE


def get_data_and_text(dataset_name, train_perc=0.6, val_perc=0.2, test_perc=0.2, use_text=True, seed=42):
    # file path
    data_path = f"dataset/{dataset_name}_data.pt"
    text_path = f"dataset/{dataset_name}_text.pkl"

    # check if the files exist
    if os.path.exists(data_path) and os.path.exists(text_path):
        print("Load data and text files...")

        # load data and text
        data = torch.load(data_path)
        with open(text_path, "rb") as f:
            text = pickle.load(f)
    else:
        print("generate data and text files...")
        # regenerate data and text
        data, text = load_data(dataset_name, train_perc, val_perc, test_perc, use_text, seed)

        # save data and text
        torch.save(data, data_path)
        with open(text_path, "wb") as f:
            pickle.dump(text, f)
        print("data and text files saved!")

    return data, text

model_name = 'graphsage'
dataset_name = 'cora'
task = 'link'

# Step 1: Load cora Dataset
data, text = get_data_and_text(dataset_name)


from sklearn.decomposition import PCA
import torch
# PCA
def apply_pca(data, target_dim=128):
    pca = PCA(n_components=target_dim)
    reduced_features = pca.fit_transform(data.x.numpy())
    data.x = torch.tensor(reduced_features, dtype=torch.float32)
    return data

# suppose `cora_data` is the loaded Cora dataset
data = apply_pca(data, target_dim=128)


if dataset_name == 'arxiv':
    edge_index_reversed = data.edge_index[[1, 0], :]  # reverse the edge direction
    edge_index_sym = torch.cat([data.edge_index, edge_index_reversed], dim=1)  # concat original and reversed edges

    edge_index_sym = torch.unique(edge_index_sym, dim=1)  # 按列去重
    data.edge_index = edge_index_sym

edge_index = data.edge_index
num_edges = edge_index.size(1)

train_node_indices = torch.tensor([i for i, x in enumerate(data.train_mask) if x])
test_node_indices = torch.tensor([i for i, x in enumerate(data.test_mask) if x])


train_mask = torch.isin(edge_index[0], train_node_indices) | torch.isin(edge_index[1], train_node_indices)
test_mask = torch.isin(edge_index[0], test_node_indices) & torch.isin(edge_index[1], test_node_indices)
train_pos_edges = edge_index[:, train_mask]
test_pos_edges = edge_index[:, test_mask]

import numpy as np

# 存在的边
existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

# 自环边
self_loops = set((i, i) for i in range(data.num_nodes))

# 初始化集合和计数器
negative_edges = set()

while len(negative_edges) < num_edges:
    # 随机生成 (u, v) 对
    u = np.random.randint(0, data.num_nodes)
    v = np.random.randint(0, data.num_nodes)

    # 检查是否为无效边 (existing_edges or self_loops)
    if (u, v) not in existing_edges and (u, v) not in self_loops and (u, v) not in negative_edges:
        negative_edges.add((u, v))

# 转换为列表（如果需要）
negative_edges = list(negative_edges)

# negative_edges = list(all_edges - existing_edges - self_loops)

# 随机选择负样本
# neg_eids = np.random.choice(len(negative_edges), size=num_edges, replace=False)
neg_u, neg_v = zip(*[negative_edges[i] for i in range(len(negative_edges))])
neg_u = torch.tensor(neg_u)  # 或者用 torch.tensor(neg_u)
neg_v = torch.tensor(neg_v)  # 或者用 torch.tensor(neg_v)

# Negative examples (non-existent edges)
train_size = train_pos_edges.size(1)
test_size = test_pos_edges.size(1)
train_neg_u, train_neg_v = neg_u[:train_size], neg_v[:train_size]
test_neg_u, test_neg_v = neg_u[num_edges-test_size:], neg_v[num_edges-test_size:]

train_edge_index = edge_index[:, train_mask]
train_data = data.clone()
train_data.edge_index = train_edge_index


def evaluate_accuracy(model, data, pos_edges, neg_u, neg_v, threshold=0.5):
    model.eval()

    # Get node features
    node_feats = data.x

    # Get embeddings from the GCN model
    embeddings = model(node_feats, data.edge_index)

    # Compute positive samples scores
    pos_score = (embeddings[pos_edges[0]] * embeddings[pos_edges[1]]).sum(dim=1)

    # Compute negative samples scores
    neg_score = (embeddings[neg_u] * embeddings[neg_v]).sum(dim=1)

    # Combine scores and ground truth labels
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(len(pos_score)), torch.zeros(len(neg_score))])

    # Apply threshold to predict labels
    predictions = (scores > threshold).float()

    # Compute accuracy
    correct = (predictions == labels).sum().item()
    accuracy = correct / len(labels)

    return accuracy


in_feats = data.x.shape[1]
h_feats = 128  # Number of hidden units
if model_name == 'gcn':
    model = GCN(in_feats, h_feats)
elif model_name == 'gat':
    model = GAT(in_feats, h_feats)
else:
    model = GraphSAGE(in_feats, h_feats, 2, 0.5)

model.load_state_dict(torch.load(f'baseline/output_model/{model_name}_{task}_arxiv_best_model.pth'))
acc = evaluate_accuracy(model, data, test_pos_edges, test_neg_u, test_neg_v)
print(f"Final ACC (Best Model): {acc}")


