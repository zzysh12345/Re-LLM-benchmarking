import os
import sys

from torch_geometric.nn import GCNConv

os.chdir('../')
print(os.getcwd())
project_root = os.getcwd()
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj, add_self_loops
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
from utils.utils import load_data


def get_data_and_text(dataset_name, train_perc=0.6, val_perc=0.2, test_perc=0.2, use_text=True, seed=42):
    # 文件路径
    data_path = f"dataset/{dataset_name}_data.pt"
    text_path = f"dataset/{dataset_name}_text.pkl"

    # 检查文件是否存在
    if os.path.exists(data_path) and os.path.exists(text_path):
        print("Load data and text files...")
        # 加载已保存的 data 和 text
        data = torch.load(data_path)
        with open(text_path, "rb") as f:
            text = pickle.load(f)
    else:
        print("generate data and text files...")
        # 重新生成 data 和 text
        data, text = load_data(dataset_name, train_perc, val_perc, test_perc, use_text, seed)

        # 保存 data 和 text
        torch.save(data, data_path)
        with open(text_path, "wb") as f:
            pickle.dump(text, f)
        print("data and text files saved!")

    return data, text

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = 'graphcl'
dataset_name = 'arxiv'
task = 'link'

data, text = get_data_and_text(dataset_name)

if dataset_name == 'arxiv':
    edge_index_reversed = data.edge_index[[1, 0], :]  # 反转边的方向
    edge_index_sym = torch.cat([data.edge_index, edge_index_reversed], dim=1)  # 拼接原始和反向边

    # 2. 去重
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

# adj = to_scipy_sparse_matrix(edge_index, num_nodes=data.num_nodes).tocoo()
# adj_neg = 1 - adj.todense() - np.eye(data.num_nodes)
#
# neg_u, neg_v = np.where(adj_neg != 0)

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

# ------------------ GraphCL Augmentations ------------------
def random_edge_drop(edge_index, drop_prob=0.2):
    """Randomly drop edges."""
    device = edge_index.device  # 确保设备一致
    mask = torch.rand(edge_index.size(1), device=device) > drop_prob  # 在相同设备上生成 mask
    return edge_index[:, mask]


def random_node_feature_masking(x, mask_prob=0.2):
    """Randomly mask node features."""
    device = x.device  # 确保设备一致
    mask = torch.rand(x.size(), device=device) > mask_prob  # 在相同设备上生成 mask
    return x * mask


def graph_augmentation(data, drop_prob=0.2, mask_prob=0.2):
    """Apply GraphCL augmentations: edge drop and feature masking."""
    edge_index = random_edge_drop(data.edge_index, drop_prob)
    x = random_node_feature_masking(data.x, mask_prob)
    return Data(x=x, edge_index=edge_index)


# ------------------ GraphCL Loss ------------------
def contrastive_loss(z1, z2, temperature=0.5):
    """Compute InfoNCE contrastive loss."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Similarity matrix
    sim_matrix = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)

    # Cross entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


# ------------------ Model Definition ------------------
# class GCN(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats):
#         super(GCN, self).__init__()
#         self.conv1 = nn.Linear(in_feats, hidden_feats)
#         self.conv2 = nn.Linear(hidden_feats, out_feats)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         return x

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, out_feats)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First layer with ReLU and dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second layer without activation
        x = self.conv2(x, edge_index)
        return x

# ------------------ Training Functions ------------------
def train_graphcl(model, data, optimizer, epochs=200, temperature=0.5):
    """GraphCL pretraining loop."""
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Apply augmentations
        data1 = graph_augmentation(data)
        data2 = graph_augmentation(data)

        # Compute embeddings
        z1 = model(data1.x, data1.edge_index)
        z2 = model(data2.x, data2.edge_index)

        # Compute contrastive loss
        loss = contrastive_loss(z1, z2, temperature)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


def train_link_prediction(model, data, pos_edges, neg_u, neg_v, optimizer):
    model.train()
    optimizer.zero_grad()

    # Get node features
    node_feats = data.x

    # Get embeddings from the GCN model
    embeddings = model(node_feats, data.edge_index)

    # Compute positive samples scores
    pos_score = (embeddings[pos_edges[0]] * embeddings[pos_edges[1]]).sum(dim=1)

    # Compute negative samples scores
    neg_score = (embeddings[neg_u] * embeddings[neg_v]).sum(dim=1)

    # Labels: 1 for positive samples, 0 for negative samples
    labels = torch.cat([torch.ones(len(pos_score)), torch.zeros(len(neg_score))])

    # Combine the scores (positive and negative)
    scores = torch.cat([pos_score, neg_score])

    # Compute binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


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


# ------------------ Main Code ------------------
# Model and optimizer
model = GCN(data.x.size(1), 256, 256)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

# Step 1: GraphCL Pretraining
print("Starting GraphCL Pretraining...")
train_graphcl(model, train_data, optimizer, epochs=300)

# Step 2: Link Prediction Fine-tuning
print("Starting Link Prediction Fine-tuning...")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Early Stopping 参数
patience = 20  # 容忍次数
min_delta = 1e-3  # 指标改善的最小变化
best_acc = 0
counter = 0

# Step 7: Training Loop with Early Stopping
for epoch in range(8000):
    # 训练模型
    loss = train_link_prediction(model, train_data, train_pos_edges, train_neg_u, train_neg_v, optimizer)
    print(f"Epoch {epoch+1}, Loss: {loss}")

    # 每 10 个 epoch 评估一次
    if (epoch + 1) % 10 == 0:
        acc = evaluate_accuracy(model, data, test_pos_edges, test_neg_u, test_neg_v)
        print(f"Epoch {epoch+1}, ACC: {acc}")

        # Early Stopping 检查
        if acc > best_acc + min_delta:
            best_acc = acc  # 更新最佳指标
            counter = 0  # 重置计数器
            # 可选：保存当前最佳模型
            torch.save(model.state_dict(), f'baseline/output_model/{model_name}_{task}_{dataset_name}_best_model.pth')
        else:
            counter += 1
            print(f"Early Stopping Counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered.")
            break

    # 更新学习率调度器
    scheduler.step()

# Step 8: 加载最佳模型进行测试
model.load_state_dict(torch.load(f'baseline/output_model/{model_name}_{task}_{dataset_name}_best_model.pth'))
acc = evaluate_accuracy(model, data, test_pos_edges, test_neg_u, test_neg_v)
print(f"Final ACC (Best Model): {acc}")