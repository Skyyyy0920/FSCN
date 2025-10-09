import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCNBaseline(nn.Module):
    """
    纯 GCN 基线模型
    仅使用结构连接（SC）矩阵通过图卷积网络进行分类
    """

    def __init__(self, num_nodes, d_model=128, num_classes=2, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.num_layers = num_layers

        # GCN 层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 第一层：从输入特征到隐藏维度
        self.convs.append(GCNConv(num_nodes, d_model))
        self.batch_norms.append(nn.BatchNorm1d(d_model))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(d_model, d_model))
            self.batch_norms.append(nn.BatchNorm1d(d_model))

        # 最后一层
        self.convs.append(GCNConv(d_model, d_model))
        self.batch_norms.append(nn.BatchNorm1d(d_model))

        self.dropout = nn.Dropout(dropout)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(num_nodes * d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, fc_matrix, sc_matrix):
        """
        Args:
            fc_matrix: [B, N, N] - 功能连接矩阵（不使用）
            sc_matrix: [B, N, N] - 结构连接矩阵
        Returns:
            logits: [B, num_classes]
        """
        B = sc_matrix.shape[0]

        # 对每个样本分别处理
        outputs = []
        for i in range(B):
            sc = sc_matrix[i]

            # 将 SC 矩阵转换为图结构
            edge_index, edge_weight = self._sc_to_edge(sc)

            # 节点特征使用 SC 矩阵本身
            x = sc  # [N, N]

            # 通过 GCN 层
            for j, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                x = conv(x, edge_index, edge_weight)
                x = bn(x)
                if j < self.num_layers - 1:
                    x = F.relu(x)
                    x = self.dropout(x)

            outputs.append(x)

        # 合并所有样本
        x = torch.stack(outputs, dim=0)  # [B, N, d_model]

        # 展平并分类
        x = x.flatten(start_dim=1)  # [B, N * d_model]
        logits = self.classifier(x)

        return logits

    def _sc_to_edge(self, sc_matrix):
        edge_index = sc_matrix.nonzero(as_tuple=False).t()
        edge_weight = sc_matrix[edge_index[0], edge_index[1]]

        # 如果没有边，添加自环
        if edge_index.shape[1] == 0:
            N = sc_matrix.shape[0]
            edge_index = torch.arange(N, device=sc_matrix.device).repeat(2, 1)
            edge_weight = torch.ones(N, device=sc_matrix.device)

        return edge_index, edge_weight


class TransformerBaseline(nn.Module):
    """
    纯 Transformer 基线模型
    仅使用功能连接（FC）矩阵通过 Transformer 进行分类
    """

    def __init__(self, num_nodes, d_model=128, num_classes=2, nhead=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_nodes,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(num_nodes, d_model),
            nn.LayerNorm(d_model)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_nodes * d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, fc_matrix, sc_matrix):
        """
        Args:
            fc_matrix: [B, N, N] - 功能连接矩阵
            sc_matrix: [B, N, N] - 结构连接矩阵（不使用）
        Returns:
            logits: [B, num_classes]
        """
        # 使用 FC 矩阵作为输入
        x = fc_matrix  # [B, N, N]

        # 通过 Transformer
        x = self.transformer(x)  # [B, N, N]

        # 投影到隐藏维度
        x = self.projection(x)  # [B, N, d_model]

        # 展平并分类
        x = x.flatten(start_dim=1)  # [B, N * d_model]
        logits = self.classifier(x)

        return logits


class GATBaseline(nn.Module):
    """
    Graph Attention Network (GAT) 基线模型
    使用注意力机制的图神经网络
    """

    def __init__(self, num_nodes, d_model=128, num_classes=2, num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.num_layers = num_layers

        # GAT 层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 第一层
        self.convs.append(GATConv(num_nodes, d_model // heads, heads=heads, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(d_model))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(d_model, d_model // heads, heads=heads, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(d_model))

        # 最后一层
        self.convs.append(GATConv(d_model, d_model, heads=1, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(d_model))

        self.dropout = nn.Dropout(dropout)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(num_nodes * d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, fc_matrix, sc_matrix):
        """
        Args:
            fc_matrix: [B, N, N]
            sc_matrix: [B, N, N]
        Returns:
            logits: [B, num_classes]
        """
        B = sc_matrix.shape[0]

        outputs = []
        for i in range(B):
            sc = sc_matrix[i]
            edge_index, edge_weight = self._sc_to_edge(sc)
            x = sc

            for j, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                x = conv(x, edge_index)
                x = bn(x)
                if j < self.num_layers - 1:
                    x = F.elu(x)
                    x = self.dropout(x)

            outputs.append(x)

        x = torch.stack(outputs, dim=0)
        x = x.flatten(start_dim=1)
        logits = self.classifier(x)

        return logits

    def _sc_to_edge(self, sc_matrix):
        edge_index = sc_matrix.nonzero(as_tuple=False).t()
        edge_weight = sc_matrix[edge_index[0], edge_index[1]]

        if edge_index.shape[1] == 0:
            N = sc_matrix.shape[0]
            edge_index = torch.arange(N, device=sc_matrix.device).repeat(2, 1)
            edge_weight = torch.ones(N, device=sc_matrix.device)

        return edge_index, edge_weight


class FCOnlyBaseline(nn.Module):
    """
    仅使用 FC 矩阵的简单 MLP 基线
    """

    def __init__(self, num_nodes, d_model=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes

        self.mlp = nn.Sequential(
            nn.Linear(num_nodes * num_nodes, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, fc_matrix, sc_matrix):
        """
        Args:
            fc_matrix: [B, N, N]
            sc_matrix: [B, N, N] (不使用)
        """
        x = fc_matrix.flatten(start_dim=1)  # [B, N*N]
        logits = self.mlp(x)
        return logits


class SCOnlyBaseline(nn.Module):
    """
    仅使用 SC 矩阵的简单 MLP 基线
    """

    def __init__(self, num_nodes, d_model=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes

        self.mlp = nn.Sequential(
            nn.Linear(num_nodes * num_nodes, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, fc_matrix, sc_matrix):
        """
        Args:
            fc_matrix: [B, N, N] (不使用)
            sc_matrix: [B, N, N]
        """
        x = sc_matrix.flatten(start_dim=1)  # [B, N*N]
        logits = self.mlp(x)
        return logits


# 模型字典，方便调用
BASELINE_MODELS = {
    'gcn': GCNBaseline,
    'transformer': TransformerBaseline,
    'gat': GATBaseline,
    'fc_only': FCOnlyBaseline,
    'sc_only': SCOnlyBaseline,
}


def get_baseline_model(model_name, num_nodes, d_model=128, num_classes=2, **kwargs):
    """
    获取基线模型
    
    Args:
        model_name: 模型名称 ('gcn', 'transformer', 'gat', 'fc_only', 'sc_only')
        num_nodes: 节点数
        d_model: 隐藏维度
        num_classes: 分类数
        **kwargs: 其他模型参数
    
    Returns:
        model: 基线模型实例
    """
    if model_name not in BASELINE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(BASELINE_MODELS.keys())}")

    model_class = BASELINE_MODELS[model_name]
    return model_class(num_nodes=num_nodes, d_model=d_model, num_classes=num_classes, **kwargs)
