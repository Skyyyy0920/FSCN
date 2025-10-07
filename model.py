import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for FC branch
    Note: without Positional Embedding
    """

    def __init__(self, input_dim, d_model, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.proj(x)
        x = self.transformer(x)
        return x


class GCNBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_weight=None):
        # x: [N, input_dim]
        # edge_index: [2, num_edges]
        # edge_weight: [num_edges]
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, Z_F, Z_S):
        # [B, N, d]
        gate = self.gate_net(torch.cat([Z_F, Z_S], dim=-1))
        Z_fused = gate * Z_F + (1 - gate) * Z_S
        return self.proj(Z_fused)


class StructureAwareCrossAttention(nn.Module):
    """
    Structure-aware Cross-Attention:
    在 cross-attention 过程中引入结构感知门控 (per-edge gate)，
    利用 SC 矩阵调制不同节点之间的信息流强度。
    """

    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5

        # Q, K, V projection
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # 可学习的门控映射 (映射结构连通强度 -> gate)
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, Q_input, KV_input, SC):
        """
        Q_input: [B, N, d_model]  — 来自 FC 或 SC 分支
        KV_input: [B, N, d_model] — 另一模态 (cross)
        SC: [B, N, N] — 结构连接矩阵 (normalized)
        """
        B, N, _ = Q_input.shape

        Q = self.Wq(Q_input).view(B, N, self.nhead, self.d_head).transpose(1, 2)  # [B, h, N, d_head]
        K = self.Wk(KV_input).view(B, N, self.nhead, self.d_head).transpose(1, 2)
        V = self.Wv(KV_input).view(B, N, self.nhead, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, h, N, N]

        g = self.gate_mlp(SC.unsqueeze(-1)).squeeze(-1)  # [B, N, N]
        g = g.unsqueeze(1)  # broadcast 到多头 [B, h, N, N]

        attn_scores = attn_scores + torch.log(g + 1e-8)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # [B, h, N, d_head]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        attn_out = self.out_proj(attn_out)

        return attn_out


class DualBranchModel(nn.Module):
    """
    Dual Branch Model：FC Branch + SC Branch + Structure-Aware Cross-Attention
    """

    def __init__(self, num_nodes, d_model=128, num_classes=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model

        # FC 分支：Transformer Encoder
        self.fc_branch = TransformerEncoder(
            input_dim=num_nodes,
            d_model=d_model,
            nhead=4,
            num_layers=2
        )

        # SC 分支：GCN
        self.sc_branch = GCNBranch(
            input_dim=num_nodes,
            hidden_dim=d_model,
            output_dim=d_model
        )

        # ===== 替换为结构感知门控交叉注意力 =====
        self.cross_attn_fc2sc = StructureAwareCrossAttention(d_model, nhead=4)
        self.cross_attn_sc2fc = StructureAwareCrossAttention(d_model, nhead=4)

        self.gate_alpha_1 = nn.Parameter(torch.tensor(0.8))
        self.gate_alpha_2 = nn.Parameter(torch.tensor(0.8))
        self.norm_fc = nn.LayerNorm(d_model)
        self.norm_sc = nn.LayerNorm(d_model)

        self.fusion_mlp = GatedFusion(d_model)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(num_nodes * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, fc_matrix, sc_matrix):
        """
        fc_matrix: [B, N, N]
        sc_matrix: [B, N, N]
        """
        B = fc_matrix.shape[0]

        Z_F = self.fc_branch(fc_matrix)  # [B, N, d_model]

        Z_S_list = []
        for i in range(B):
            sc = sc_matrix[i]
            edge_index, edge_weight = self._sc_to_edge(sc)
            node_features = sc
            z_s = self.sc_branch(node_features, edge_index, edge_weight)
            Z_S_list.append(z_s)
        Z_S = torch.stack(Z_S_list, dim=0)  # [B, N, d_model]

        CA_F2S = self.cross_attn_fc2sc(Z_F, Z_S, sc_matrix)  # gated by SC
        CA_S2F = self.cross_attn_sc2fc(Z_S, Z_F, sc_matrix)

        Z_F_fused = self.norm_fc(Z_F + self.gate_alpha_1 * CA_F2S)
        Z_S_fused = self.norm_sc(Z_S + self.gate_alpha_2 * CA_S2F)

        Z_joint = self.fusion_mlp(Z_F_fused, Z_S_fused)

        logits = self.classifier(Z_joint.flatten(start_dim=1))

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
