import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv
from torch.nn import LayerNorm, Linear


# class GraphSAGE(nn.Module):
#     def __init__(self, in_feats, h_feats):
#         super(GraphSAGE, self).__init__()
#         self.conv1 = SAGEConv(in_feats, h_feats)
#         self.conv2 = SAGEConv(h_feats, h_feats)
#
#     def forward(self, x, edge_index):
#         h = self.conv1(x, edge_index)
#         h = F.relu(h)
#         h = self.conv2(h, edge_index)
#         return h


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers, dropout):
        super(GraphSAGE, self).__init__()

        # 定义 GraphSAGE 的卷积层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第一层
        self.convs.append(SAGEConv(in_feats, hidden_feats))
        self.bns.append(nn.BatchNorm1d(hidden_feats))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_feats, hidden_feats))
            self.bns.append(nn.BatchNorm1d(hidden_feats))

        # 最后一层
        self.convs.append(SAGEConv(hidden_feats, in_feats))

        self.dropout = dropout

    def reset_parameters(self):
        # 重置参数
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        # 遍历所有卷积层
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)  # SAGE 卷积
            x = self.bns[i](x)  # 批归一化
            x = F.relu(x)  # 激活函数
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

        # 最后一层，不需要批归一化和 Dropout
        x = self.convs[-1](x, edge_index)
        return x


import torch.nn as nn
from torch_geometric.nn import SAGEConv



import torch.nn as nn
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()

        # 定义卷积层和批归一化层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 输入层
        self.convs.append(GCNConv(in_feats, h_feats))
        self.bns.append(nn.BatchNorm1d(h_feats))

        # 隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(h_feats, h_feats))
            self.bns.append(nn.BatchNorm1d(h_feats))

        # 输出层
        self.convs.append(GCNConv(h_feats, h_feats))

        self.dropout = dropout

    def reset_parameters(self):
        # 重置参数
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        # 前向传播
        for i, conv in enumerate(self.convs[:-1]):  # 遍历前 num_layers - 1 层
            x = conv(x, edge_index)  # GCN 卷积
            x = self.bns[i](x)  # 批归一化
            x = F.relu(x)  # 激活函数
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

        x = self.convs[-1](x, edge_index)  # 最后一层不需要激活和 dropout
        return x

class GIN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GIN, self).__init__()

        # 第一个 GINConv 层
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats),
            nn.ReLU()
        ))

        # 第二个 GINConv 层
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats),
            nn.ReLU()
        ))

    def forward(self, x, edge_index):
        # 第一个 GIN 层
        h = self.conv1(x, edge_index)
        h = F.relu(h)

        # 第二个 GIN 层
        h = self.conv2(h, edge_index)
        return h


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv



class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, heads=1):
        super(GAT, self).__init__()
        # 第一层 GATConv
        self.conv1 = GATConv(in_feats, h_feats, heads=heads, concat=True)
        # 第二层 GATConv
        self.conv2 = GATConv(h_feats * heads, h_feats, heads=1, concat=False)

    def forward(self, x, edge_index):
        # 第一层 GATConv + 激活函数
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        # 第二层 GATConv
        h = self.conv2(h, edge_index)
        return h




import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm



class GCN_node(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN_node, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class GNNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, ln, gnn='gcn'):
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)
        if gnn=='gcn':
            self.conv = GCNConv(in_channels, out_channels)
        elif gnn=='sage':
            self.conv = SAGEConv(in_channels, out_channels)
        elif gnn=='gat':
            self.conv = GATConv(in_channels, out_channels)
        self.dropout = dropout
        self.ln = ln
    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        if self.ln:
            x = self.norm(x).relu()
        else:
            x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv(x, edge_index)

class GNN_new(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5,
                 dropout=0.5, ln=True, jk=True, res=True):
        super().__init__()

        self.dropout = dropout
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.norm = LayerNorm(hidden_channels, elementwise_affine=True)
        self.ln = ln
        self.jk = jk
        self.res = res
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GNNConv(
                hidden_channels,
                hidden_channels,
                dropout,
                self.ln,
                gnn='gcn'
            )
            self.convs.append(conv)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x_final = 0
        x = self.lin1(x)
        x_final += x
        for (conv) in self.convs:
            if self.res:
                x = conv(x, edge_index) + x
            else:
                x = conv(x, edge_index)
            x_final += x
        if self.ln:
            x = self.norm(x).relu()
        else:
            x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.jk:
            x = x_final
        else:
            pass

        return self.lin2(x).log_softmax(dim=-1)

class SAGE_node(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE_node, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


from torch.nn import Linear, BatchNorm1d, ModuleList, ReLU, Dropout
from torch_geometric.nn import GINConv

class GIN_node(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GIN_node, self).__init__()

        self.convs = ModuleList()
        self.bns = ModuleList()

        # 输入层
        self.convs.append(
            GINConv(
                torch.nn.Sequential(
                    Linear(in_channels, hidden_channels),
                    ReLU(),
                    Linear(hidden_channels, hidden_channels),
                )
            )
        )
        self.bns.append(BatchNorm1d(hidden_channels))

        # 隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(
                    torch.nn.Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                    )
                )
            )
            self.bns.append(BatchNorm1d(hidden_channels))

        # 输出层
        self.convs.append(
            GINConv(
                torch.nn.Sequential(
                    Linear(hidden_channels, out_channels)
                )
            )
        )

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)


from torch_geometric.nn import GATConv

class GAT_node(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                  dropout, heads=2):
        super(GAT_node, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # 第一层
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))

        # 最后一层
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)