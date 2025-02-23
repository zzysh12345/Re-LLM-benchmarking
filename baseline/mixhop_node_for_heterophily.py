import os
import sys

# from LLM.llama.Llama_8B_Instruct_cora_node_5shot_no_link import model

os.chdir('../')
print(os.getcwd())
project_root = os.getcwd()
sys.path.append(project_root)

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import pickle
from baseline.model import GCN_node, GAT_node, SAGE_node


# from ogb.nodeproppred import PygNodePropPredDataset

def get_data_and_text(dataset_name, drop_rate, is_random_drop, train_perc=0.6, val_perc=0.2, test_perc=0.2, use_text=True, seed=42):
    # 文件路径
    # text_path = f"dataset/{dataset_name}_text.pkl"
    name_sign = int(drop_rate * 100)
    if is_random_drop:
        data_path = f"dataset/random_{name_sign}_drop/{dataset_name}_data.pt"
    else:
        data_path = f"dataset/{name_sign}_drop/{dataset_name}_data.pt"

    print(data_path)
    # 检查文件是否存在
    if os.path.exists(data_path):
        print("Load data file...")
        # 加载已保存的 data 和 text
        data = torch.load(data_path)

    else:
        print("No such files!")
        return

    return data

# 1. 加载Cora数据集
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = 'mixhop'
dataset_name = 'arxiv'
task = 'node'
drop_rate = 1
is_random_drop = False

name_sign = int(drop_rate * 100)
if is_random_drop:
    sign = 'random_drop'
else:
    sign = 'drop'

# Load cora Dataset
data = get_data_and_text(dataset_name, drop_rate, is_random_drop)
if data.y.dim() == 2:
    data.y = data.y.view(-1)

# mask = data.edge_index[0] != data.edge_index[1]  # 筛选出起始节点和目标节点不相等的边
# data.edge_index = data.edge_index[:, mask]  # 应用掩码，保留非自环边
# 3. 设置训练设备
in_feats = data.x.shape[1]
num_classes = data.y.unique().size(0)
h_feats = 256  # Number of hidden units
# model = EnhancedGCN(in_channels=in_feats, hidden_channels=h_feats, out_channels=num_classes).to(device)
from torch_geometric.nn import MixHopConv
class MixHopNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=512):
        super().__init__()
        # 第一层 MixHopConv：聚合 0-hop, 1-hop, 2-hop 特征
        self.conv1 = MixHopConv(
            in_channels=num_features,
            out_channels=hidden_channels,
            powers=[0, 1, 2],  # 混合不同跳数特征
            add_self_loops=True  # 是否添加自环
        )
        # 第二层 MixHopConv：进一步聚合特征
        self.conv2 = MixHopConv(
            in_channels=hidden_channels * 3,  # 输入维度 = 上一层的输出维度（3 是 len(powers)）
            out_channels=num_classes,
            powers=[0, 1],  # 简化跳数组合
            add_self_loops=True
        )
        self.dropout = torch.nn.Dropout(0.6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

model = MixHopNet(num_features=in_feats, num_classes=num_classes).to(device)
data.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 4. 训练函数
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# 5. 测试函数
def evaluate_accuracy(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # 取预测的类别

    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())

    return acc

# 6. 训练和评估
patience = 20  # 容忍次数
min_delta = 1e-3  # 指标改善的最小变化
best_acc = 0
counter = 0
for epoch in range(8000):
    # 训练模型
    loss = train(model, data, optimizer)
    print(f"Epoch {epoch+1}, Loss: {loss}")

    if (epoch + 1) % 10 == 0:
        acc = evaluate_accuracy(model, data)
        print(f"Epoch {epoch+1}, ACC: {acc}")

        # Early Stopping 检查
        if acc > best_acc + min_delta:
            best_acc = acc  # 更新最佳指标
            counter = 0  # 重置计数器
            # 可选：保存当前最佳模型
            torch.save(model.state_dict(), f'baseline/output_model/{model_name}_{sign}_{name_sign}_{task}_{dataset_name}_best_model.pth')
        else:
            counter += 1
            print(f"Early Stopping Counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered.")
            break

    # 更新学习率调度器
    scheduler.step()

model.load_state_dict(torch.load(f'baseline/output_model/{model_name}_{sign}_{name_sign}_{task}_{dataset_name}_best_model.pth'))
acc = evaluate_accuracy(model, data)
print(f"Final ACC (Best Model): {acc}")

