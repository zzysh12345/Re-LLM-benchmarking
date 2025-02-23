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
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
import pickle
import torch_geometric.transforms as T
import numpy as np


# === 加载 Cora 数据集 ===

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
        # data, text = load_data(dataset_name, train_perc, val_perc, test_perc, use_text, seed)
        #
        # # 保存 data 和 text
        # torch.save(data, data_path)
        # with open(text_path, "wb") as f:
        #     pickle.dump(text, f)
        # print("data and text files saved!")

    return data, text

# 1. 加载Cora数据集
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = 'graphcl'
dataset_name = 'pubmed'
task = 'node'
shots_per_class = 10

# Load cora Dataset
data, text = get_data_and_text(dataset_name)
if dataset_name == 'arxiv':
    edge_index_reversed = data.edge_index[[1, 0], :]  # 反转边的方向
    edge_index_sym = torch.cat([data.edge_index, edge_index_reversed], dim=1)  # 拼接原始和反向边

    # 2. 去重
    edge_index_sym = torch.unique(edge_index_sym, dim=1)  # 按列去重
    data.edge_index = edge_index_sym

if data.y.dim() == 2:
    data.y = data.y.view(-1)

num_classes = data.y.unique().size(0)

import random
def create_few_shot_mask(data, shots_per_class, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    train_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)

    for c in range(num_classes):
        class_indices = torch.where(data.y[data.train_mask] == c)[0]
        selected_indices = random.sample(class_indices.tolist(), min(shots_per_class, len(class_indices)))
        train_mask[selected_indices] = True

    return train_mask

few_shot_train_mask = create_few_shot_mask(data, shots_per_class=shots_per_class)

# 更新数据掩码
data.train_mask = few_shot_train_mask


# === 定义 GNN 模型 ===
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# === 数据增强方法 ===
def feature_masking(x, mask_rate=0.3):
    """随机掩盖特征（节点属性数据增强）"""
    # mask = torch.rand(x.size()) > mask_rate
    mask = (torch.rand(x.size(), device=x.device) > mask_rate)  # 将 mask 创建在 x 所在的设备上
    return x * mask


# def edge_perturbation(edge_index, num_nodes, perturb_rate=0.1):
#     """边的扰动（图结构数据增强）"""
#     num_edges = edge_index.size(1)
#     num_perturb = int(num_edges * perturb_rate)
#
#     # 随机移除边
#     perm = torch.randperm(num_edges, device=device)[:num_perturb]  # 在相同设备生成随机索引
#     edge_index = edge_index[:, perm]
#
#     # 随机添加边
#     new_edges = torch.randint(0, num_nodes, (2, num_perturb), device=device)  # 在相同设备生成随机边
#     edge_index = torch.cat([edge_index, new_edges], dim=1)
#     return edge_index

from torch_sparse import SparseTensor

def edge_perturbation(edge_index: SparseTensor, num_nodes: int, perturb_rate: float = 0.1):
    """边的扰动（图结构数据增强）适用于 SparseTensor 格式"""
    # 转换为 COO 格式以便操作
    row, col, value = edge_index.coo()
    num_edges = row.size(0)
    num_perturb = int(num_edges * perturb_rate)

    # 随机移除边
    perm = torch.randperm(num_edges, device=row.device)[:num_perturb]  # 在相同设备生成随机索引
    row, col = row[perm], col[perm]

    # 随机添加边
    new_edges_row = torch.randint(0, num_nodes, (num_perturb,), device=row.device)  # 在相同设备生成随机起点
    new_edges_col = torch.randint(0, num_nodes, (num_perturb,), device=row.device)  # 在相同设备生成随机终点

    # 合并边（包括移除后保留的边和新添加的边）
    row = torch.cat([row, new_edges_row], dim=0)
    col = torch.cat([col, new_edges_col], dim=0)

    # 创建新的 SparseTensor
    new_edge_index = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    return new_edge_index


# === 对比损失 ===
def contrastive_loss(z1, z2, temperature=0.5):
    """实现对比损失（GraphCL 核心）"""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    sim_matrix = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# def contrastive_loss(z1, z2, temperature=0.5, chunk_size=512):
#     """实现对比损失（优化显存占用）"""
#     z1 = F.normalize(z1, dim=1)
#     z2 = F.normalize(z2, dim=1)
#
#     # 分块计算相似性矩阵
#     sim_matrix = []
#     for i in range(0, z1.size(0), chunk_size):
#         sim_chunk = torch.mm(z1[i:i+chunk_size], z2.T) / temperature
#         sim_matrix.append(sim_chunk)
#     sim_matrix = torch.cat(sim_matrix, dim=0)
#
#     labels = torch.arange(z1.size(0)).to(z1.device)
#     loss = F.cross_entropy(sim_matrix, labels)
#     return loss


# === 训练 GraphCL ===
in_feats = data.x.shape[1]
num_classes = data.y.unique().size(0)
h_feats = 256  # Number of hidden units
model = GNN(in_channels=in_feats, hidden_channels=128, out_channels=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# data = data.to(device)
transform = T.Compose([T.ToDevice(device), T.ToSparseTensor()])
data = transform(data)

# 对比学习的训练循环
epochs = 100
for epoch in range(epochs):
    model.train()

    # 原始数据编码
    z = model(data.x, data.adj_t)

    # 数据增强生成两种视图
    x_aug1 = feature_masking(data.x)
    edge_index_aug2 = edge_perturbation(data.adj_t, num_nodes=data.num_nodes)

    # 两种视图的编码
    z1 = model(x_aug1, data.adj_t)
    z2 = model(data.x, edge_index_aug2)

    # 计算对比损失
    loss = contrastive_loss(z1, z2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


# === 冷启动的节点分类任务 ===
# 冻结 GNN 的权重，只训练线性分类器
class LinearClassifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.linear(x)


classifier = LinearClassifier(in_channels=128, out_channels=num_classes).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

# 冻结 GNN 并获取节点嵌入
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.adj_t)

# 训练分类器
epochs = 50
for epoch in range(epochs):
    classifier.train()

    logits = classifier(embeddings)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


# === 测试分类器 ===
classifier.eval()
with torch.no_grad():
    logits = classifier(embeddings)
    pred = logits.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    print(f"Test Accuracy: {acc:.4f}")