import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [batch, N, input_dim]
        x = self.input_proj(x)
        x = self.transformer(x)
        return x


class GCNBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, edge_weight=None):
        # x: [N, input_dim]
        # edge_index: [2, num_edges]
        # edge_weight: [num_edges]
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class CrossAttention(nn.Module):
    """交叉注意力模块"""

    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # query, key_value: [batch, N, d_model]
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        output = self.norm(query + self.dropout(attn_output))
        return output


class DualBranchModel(nn.Module):
    """
    Dual Branch Model：FC Branch + SC Branch + Cross-Attention + OCREAD
    """

    def __init__(self, num_nodes, d_model=128, num_classes=2):
        """
        num_nodes: ROI nums
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model

        self.fc_branch = TransformerEncoder(
            input_dim=num_nodes,
            d_model=d_model,
            nhead=4,
            num_layers=2
        )

        self.sc_branch = GCNBranch(
            input_dim=num_nodes,
            hidden_dim=d_model,
            output_dim=d_model
        )

        self.cross_attn_fc2sc = CrossAttention(d_model, nhead=4)
        self.cross_attn_sc2fc = CrossAttention(d_model, nhead=4)

        self.classifier = nn.Sequential(
            nn.Linear(num_nodes * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, fc_matrix, sc_matrix):
        """
        fc_matrix: [batch, N, N]
        sc_matrix: [batch, N, N]
        """
        batch_size = fc_matrix.shape[0]

        Z_F = self.fc_branch(fc_matrix)  # [batch, N, d_model]

        Z_S_list = []
        for i in range(batch_size):
            sc = sc_matrix[i]  # [N, N]
            edge_index, edge_weight = self._sc_to_edge(sc)
            node_features = sc  # [N, N]

            z_s = self.sc_branch(node_features, edge_index, edge_weight)  # [N, d_model]
            Z_S_list.append(z_s)

        Z_S = torch.stack(Z_S_list, dim=0)  # [batch, N, d_model]

        Z_F_cross = self.cross_attn_fc2sc(Z_F, Z_S)  # FC作为query, SC作为key/value
        Z_S_cross = self.cross_attn_sc2fc(Z_S, Z_F)  # SC作为query, FC作为key/value

        Z_joint = (Z_F_cross + Z_S_cross) / 2  # [batch, N, d_model]
        Z_joint_flat = Z_joint.flatten(start_dim=1)  # [batch, N * d_model]

        logits = self.classifier(Z_joint_flat)  # [batch, num_classes]

        return logits

    def _sc_to_edge(self, sc_matrix):
        """将SC矩阵转换为边索引和边权重"""
        # 找到非零边
        edge_index = sc_matrix.nonzero(as_tuple=False).t()  # [2, num_edges]
        edge_weight = sc_matrix[edge_index[0], edge_index[1]]  # [num_edges]

        # 如果没有边，添加自环
        if edge_index.shape[1] == 0:
            N = sc_matrix.shape[0]
            edge_index = torch.arange(N, device=sc_matrix.device).repeat(2, 1)
            edge_weight = torch.ones(N, device=sc_matrix.device)

        return edge_index, edge_weight
