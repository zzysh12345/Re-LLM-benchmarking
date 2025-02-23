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


# === 加载 Cora 数据集 ===

def get_data_and_text(dataset_name, train_perc=0.6, val_perc=0.2, test_perc=0.2, use_text=True, seed=42):
    data_path = f"dataset/{dataset_name}_data.pt"
    text_path = f"dataset/{dataset_name}_text.pkl"

    if os.path.exists(data_path) and os.path.exists(text_path):
        print("Load data and text files...")
        data = torch.load(data_path)
        with open(text_path, "rb") as f:
            text = pickle.load(f)
    else:
        print("Generate data and text files...")
        # 这里需要补充生成代码
        pass

    return data, text

# === 初始化 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dataset_name = 'arxiv'

# 加载数据集
data, text = get_data_and_text(dataset_name)
if dataset_name == 'arxiv':
    edge_index_reversed = data.edge_index[[1, 0], :]
    edge_index_sym = torch.cat([data.edge_index, edge_index_reversed], dim=1)
    edge_index_sym = torch.unique(edge_index_sym, dim=1)
    data.edge_index = edge_index_sym

if data.y.dim() == 2:
    data.y = data.y.view(-1)

# === 定义 GraphMAE 模型 ===
# class GraphMAE(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GraphMAE, self).__init__()
#         self.encoder = GCNConv(in_channels, hidden_channels)
#         self.decoder = nn.Linear(hidden_channels, in_channels)
#
#     def forward(self, x, edge_index, mask):
#         z = self.encoder(x, edge_index)
#         z = F.relu(z)
#         x_reconstructed = self.decoder(z)
#         loss = F.mse_loss(x_reconstructed[mask], x[mask])  # 重建损失
#         return loss, z
# === 定义更复杂的 GraphMAE 模型 ===
# class GraphMAE(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
#         """
#         GraphMAE 模型
#         :param in_channels: 输入特征维度
#         :param hidden_channels: 隐藏层维度
#         :param out_channels: 输出嵌入维度
#         :param num_layers: 编码器层数
#         :param dropout: dropout 概率
#         """
#         super(GraphMAE, self).__init__()
#         self.num_layers = num_layers
#         self.encoder_layers = nn.ModuleList()
#         self.decoder_layers = nn.ModuleList()
#
#         # 构造编码器（多层 GCN）
#         for i in range(num_layers):
#             if i == 0:
#                 self.encoder_layers.append(GCNConv(in_channels, hidden_channels))
#             else:
#                 self.encoder_layers.append(GCNConv(hidden_channels, hidden_channels))
#
#         # 构造解码器（全连接层 + 激活函数）
#         for i in range(num_layers):
#             if i == 0:
#                 self.decoder_layers.append(nn.Linear(hidden_channels, hidden_channels))
#             elif i == num_layers - 1:
#                 self.decoder_layers.append(nn.Linear(hidden_channels, in_channels))
#             else:
#                 self.decoder_layers.append(nn.Linear(hidden_channels, hidden_channels))
#
#         self.dropout = dropout
#
#     def forward(self, x, edge_index, mask):
#         # === 编码阶段 ===
#         z = x  # 输入特征
#         for layer in self.encoder_layers:
#             z = layer(z, edge_index)  # 图卷积
#             z = F.relu(z)  # 激活函数
#             z = F.dropout(z, p=self.dropout, training=self.training)  # Dropout
#
#         # === 解码阶段 ===
#         x_reconstructed = z  # 编码输出作为解码输入
#         for i, layer in enumerate(self.decoder_layers):
#             x_reconstructed = layer(x_reconstructed)  # 全连接层
#             if i < len(self.decoder_layers) - 1:
#                 x_reconstructed = F.relu(x_reconstructed)  # 激活函数（最后一层不使用激活）
#
#         # === 重建损失 ===
#         loss = F.mse_loss(x_reconstructed[mask], x[mask])  # 仅计算掩码节点的重建损失
#
#         return loss, z
# === 数据增强：特征掩盖 ===

class GraphMAE(nn.Module):

    EPS = 1e-15

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3,
                 dropout=0.5, mask_ratio=0.5, replace_ratio=0.0):
        """
        GraphMAE 模型
        :param in_channels: 输入特征维度
        :param hidden_channels: 隐藏层维度
        :param out_channels: 输出嵌入维度
        :param num_layers: 编码器层数
        :param dropout: dropout 概率
        :param mask_ratio: 节点掩码比例
        :param replace_ratio: 噪声替换比例
        """
        super(GraphMAE, self).__init__()
        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio

        # === 编码器 ===
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.encoder_layers.append(GCNConv(in_channels, hidden_channels))
            else:
                self.encoder_layers.append(GCNConv(hidden_channels, hidden_channels))

        # === 解码器 ===
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.decoder_layers.append(nn.Linear(hidden_channels, in_channels))
            else:
                self.decoder_layers.append(nn.Linear(hidden_channels, hidden_channels))

        # 掩码替换的特殊 token
        self.encoder_mask_token = nn.Parameter(torch.zeros(1, in_channels))

        # Dropout 概率
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # === 掩码输入特征 ===
        mask_x, mask_nodes = self.encoding_mask(x)

        # === 编码阶段 ===
        z = mask_x  # 掩码后的输入特征
        for layer in self.encoder_layers:
            z = layer(z, edge_index, edge_weight)  # 图卷积
            z = F.relu(z)  # 激活函数
            z = F.dropout(z, p=self.dropout, training=self.training)  # Dropout

        # === 解码阶段 ===
        x_reconstructed = z
        for i, layer in enumerate(self.decoder_layers):
            x_reconstructed = layer(x_reconstructed)  # 全连接层
            if i < len(self.decoder_layers) - 1:
                x_reconstructed = F.relu(x_reconstructed)  # 激活函数（最后一层无激活）

        # 返回掩码节点的原始和重建特征
        return x[mask_nodes], x_reconstructed[mask_nodes]

    def encoding_mask(self, x):
        num_nodes = x.size(0)

        # 随机选择掩码节点
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(self.mask_ratio * num_nodes)
        mask_nodes = perm[:num_mask_nodes]

        out_x = x.clone()
        if self.replace_ratio > 0.0:
            num_noise_nodes = int(self.replace_ratio * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[:num_mask_nodes - num_noise_nodes]]
            noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x[mask_nodes] = 0.0

        # 为掩码节点添加特殊 token
        out_x[mask_nodes] += self.encoder_mask_token

        return out_x, mask_nodes

    def get_loss(self, x, edge_index, edge_weight=None, batch=None):
        mask_x, recon_x = self.forward(x, edge_index, edge_weight, batch)
        loss = F.mse_loss(mask_x, recon_x)  # 重建误差
        return loss

    def embed(self, x, edge_index, edge_weight=None, batch=None):
        z = x
        for layer in self.encoder_layers:
            z = layer(z, edge_index, edge_weight)
            z = F.relu(z)
        return z
def mask_features(x, mask_rate=0.3):
    mask = torch.rand(x.size(0), device=x.device) < mask_rate
    return mask

# === 训练 GraphMAE ===
in_feats = data.x.shape[1]
num_classes = data.y.unique().size(0)
hidden_feats = 256

# model = GraphMAE(in_channels=in_feats, hidden_channels=hidden_feats, out_channels=num_classes).to(device)
model = GraphMAE(in_channels=in_feats, hidden_channels=hidden_feats, out_channels=num_classes, num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

transform = T.Compose([T.ToDevice(device), T.ToSparseTensor()])
data = transform(data)

epochs = 200
for epoch in range(epochs):
    model.train()

    # 掩盖特征
    feature_mask = mask_features(data.x, mask_rate=0.3)

    # 前向传播
    loss = model.get_loss(data.x, data.adj_t)
    # loss, _ = model(data.x, data.adj_t, feature_mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# === 冷启动节点分类任务 ===
class LinearClassifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.linear(x)


classifier = LinearClassifier(in_channels=hidden_feats, out_channels=num_classes).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

# 冻结 GraphMAE 并获取节点嵌入
# model.eval()
# with torch.no_grad():
#     _, embeddings = model(data.x, data.adj_t, mask=torch.zeros_like(data.x, dtype=torch.bool))
model.eval()
with torch.no_grad():
    embeddings = model.embed(data.x, data.adj_t)
# 训练分类器
epochs = 100
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