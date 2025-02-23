import os
import sys

from torch_geometric.utils import negative_sampling

os.chdir('../')
print(os.getcwd())
project_root = os.getcwd()
sys.path.append(project_root)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# from ogb.nodeproppred import PygNodePropPredDataset
import os
import pickle
# from utils.utils import load_data
from model import GAT, GCN, GraphSAGE


# from negative_sampling import negative_sampling


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

        # 保存 data 和 text
        # torch.save(data, data_path)
        # with open(text_path, "wb") as f:
        #     pickle.dump(text, f)
        # print("data and text files saved!")

    return data, text

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = 'gat'
dataset_name = 'product'
task = 'link'

# Load cora Dataset
data, text = get_data_and_text(dataset_name)

# 原始特征维度
original_dim = data.x.shape[1]  # 100
target_dim = 128  # 目标维度

# 计算需要填充的零列数
padding_dim = target_dim - original_dim

# 使用 torch.cat 在特征后面填充零
if padding_dim > 0:
    zero_padding = torch.zeros((data.x.shape[0], padding_dim))
    data.x = torch.cat([data.x, zero_padding], dim=1)


true_indices = torch.where(data.test_mask)[0]
retain_count = max(1, int(len(true_indices) * 0.1))
retain_indices = torch.randperm(len(true_indices))[:retain_count]
new_test_mask = torch.clone(data.test_mask)  # 克隆原始 test_mask
new_test_mask[true_indices] = False  # 将所有 True 设置为 False
new_test_mask[true_indices[retain_indices]] = True  # 恢复保留的 True
data.test_mask = new_test_mask


true_indices = torch.where(data.train_mask)[0]
retain_count = max(1, int(len(true_indices) * 0.2))
retain_indices = torch.randperm(len(true_indices))[:retain_count]
new_train_mask = torch.clone(data.train_mask)  # 克隆原始 test_mask
new_train_mask[true_indices] = False  # 将所有 True 设置为 False
new_train_mask[true_indices[retain_indices]] = True  # 恢复保留的 True
data.train_mask = new_train_mask

from torch_geometric.loader import NeighborLoader

# 创建 NeighborLoader
train_loader = NeighborLoader(
    data,
    num_neighbors=[35, 20],  # 每层采样 15 和 10 个邻居
    batch_size=1024*8,         # 每批次包含 1024 个目标节点
    input_nodes=data.train_mask  # 仅采样训练集中的节点
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[35, 20],  # 同样的邻居采样设置
    batch_size=1024*8,         # 批次大小
    input_nodes=data.test_mask  # 仅采样测试集中的节点
)


def evaluate_with_neighbor_loader(model, loader, device, threshold=0.5):
    """
    使用邻居采样的 loader 评估模型的准确率。

    参数:
    - model: 图神经网络模型。
    - loader: 邻居采样的迷你批次加载器。
    - device: 计算设备（如 'cuda' 或 'cpu'）。
    - threshold: 用于正负分类的阈值，默认为 0.5。

    返回:
    - accuracy: 模型在 loader 上的准确率。
    """
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for batch in loader:
            batch = batch.to(device)

            # 前向传播获取节点嵌入
            embeddings = model(batch.x, batch.edge_index)

            # 正样本边的分数
            pos_score = (embeddings[batch.edge_index[0]] * embeddings[batch.edge_index[1]]).sum(dim=1)

            # 生成负样本边
            neg_edge_index = negative_sampling(
                edge_index=batch.edge_index,
                num_nodes=batch.num_nodes,
                num_neg_samples=batch.edge_index.size(1)
            )
            neg_score = (embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]]).sum(dim=1)

            # 合并正负样本的分数和标签
            scores = torch.cat([pos_score, neg_score]).to(device)
            labels = torch.cat([
                torch.ones(pos_score.size(0)),  # 正样本标签为 1
                torch.zeros(neg_score.size(0))  # 负样本标签为 0
            ]).to(device)

            # 根据阈值生成预测标签
            predictions = (scores > threshold).float()

            # 统计正确预测数量
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total  # 返回准确率


in_feats = data.x.shape[1]
h_feats = 128  # Number of hidden units
if model_name == 'gcn':
    model = GCN(in_feats, h_feats)
elif model_name == 'gat':
    model = GAT(in_feats, h_feats)
else:
    model = GraphSAGE(in_feats, h_feats, 2, 0.5)

model.load_state_dict(torch.load(f'baseline/output_model/{model_name}_{task}_arxiv_best_model.pth'))
model.to(device)
final_acc = evaluate_with_neighbor_loader(model, test_loader, device)
print(f"Final ACC (Best Model): {final_acc}")


