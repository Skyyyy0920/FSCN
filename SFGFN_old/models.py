"""
脑网络多模态融合模型
包含三种融合方法：
- Method A: 并行双图编码器 + 门控融合
- Method B: 多重图神经网络（消息传递求和）
- Method C: 结构连接作为骨干 + 功能连接调制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch


class TemporalEncoder(nn.Module):
    """
    时间编码器：将 fMRI 时间序列编码为节点嵌入
    注意：由于数据集只提供了连接矩阵，这里使用简单的 MLP 处理节点特征
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TemporalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Args:
            x: [N, input_dim] 节点特征
        Returns:
            [N, output_dim] 编码后的节点嵌入
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GraphPooling(nn.Module):
    """
    图级别池化层
    支持全局平均池化和注意力池化
    """
    def __init__(self, hidden_dim, pooling_type='mean'):
        super(GraphPooling, self).__init__()
        self.pooling_type = pooling_type
        
        if pooling_type == 'attention':
            self.att_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, batch):
        """
        Args:
            x: [N, hidden_dim] 节点嵌入
            batch: [N] 批次索引
        Returns:
            [batch_size, hidden_dim] 图级别嵌入
        """
        if self.pooling_type == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling_type == 'sum':
            return global_add_pool(x, batch)
        elif self.pooling_type == 'attention':
            # 计算注意力权重
            att_weights = torch.sigmoid(self.att_fc(x))  # [N, 1]
            x_weighted = x * att_weights  # [N, hidden_dim]
            return global_add_pool(x_weighted, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class Predictor(nn.Module):
    """
    预测头：MLP 分类器/回归器
    """
    def __init__(self, input_dim, hidden_dim, output_dim, task_type='classification'):
        super(Predictor, self).__init__()
        self.task_type = task_type
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] 图级别嵌入
        Returns:
            [batch_size, output_dim] 预测输出
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


# ==================== 融合方法 A: 并行双图编码器 + 门控融合 ====================

class FusionMethodA(nn.Module):
    """
    方法 A: 并行双图编码器 + 门控融合
    
    流程:
    1. 在 SC 和 FC 上分别应用 GNN
    2. 使用门控机制融合两个表示
    """
    def __init__(self, node_feature_dim, hidden_dim, num_layers=2):
        super(FusionMethodA, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # SC 分支的 GNN 层
        self.sc_convs = nn.ModuleList()
        self.sc_convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.sc_convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # FC 分支的 GNN 层
        self.fc_convs = nn.ModuleList()
        self.fc_convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.fc_convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 门控机制
        self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Batch Normalization
        self.bn_sc = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.bn_fc = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index_sc, edge_weight_sc, edge_index_fc, edge_weight_fc):
        """
        Args:
            x: [N, node_feature_dim] 节点特征
            edge_index_sc: [2, E_sc] SC 边索引
            edge_weight_sc: [E_sc] SC 边权重
            edge_index_fc: [2, E_fc] FC 边索引
            edge_weight_fc: [E_fc] FC 边权重
        Returns:
            [N, hidden_dim] 融合后的节点嵌入
        """
        # SC 分支
        h_sc = x
        for i, conv in enumerate(self.sc_convs):
            h_sc = conv(h_sc, edge_index_sc, edge_weight_sc)
            h_sc = self.bn_sc[i](h_sc)
            h_sc = F.relu(h_sc)
            h_sc = self.dropout(h_sc)
        
        # FC 分支
        h_fc = x
        for i, conv in enumerate(self.fc_convs):
            h_fc = conv(h_fc, edge_index_fc, edge_weight_fc)
            h_fc = self.bn_fc[i](h_fc)
            h_fc = F.relu(h_fc)
            h_fc = self.dropout(h_fc)
        
        # 门控融合
        h_concat = torch.cat([h_sc, h_fc], dim=-1)  # [N, 2*hidden_dim]
        gate = torch.sigmoid(self.gate_fc(h_concat))  # [N, hidden_dim]
        
        h_fused = gate * h_sc + (1 - gate) * h_fc  # [N, hidden_dim]
        
        return h_fused


# ==================== 融合方法 B: 多重图神经网络 ====================

class FusionMethodB(nn.Module):
    """
    方法 B: 多重图神经网络 (Multiplex GNN)
    
    流程:
    在每一层，同时在 SC 和 FC 上传播消息，然后求和
    """
    def __init__(self, node_feature_dim, hidden_dim, num_layers=2):
        super(FusionMethodB, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 多重图卷积层
        self.sc_convs = nn.ModuleList()
        self.fc_convs = nn.ModuleList()
        
        # 第一层
        self.sc_convs.append(GCNConv(node_feature_dim, hidden_dim))
        self.fc_convs.append(GCNConv(node_feature_dim, hidden_dim))
        
        # 后续层
        for _ in range(num_layers - 1):
            self.sc_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.fc_convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch Normalization
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index_sc, edge_weight_sc, edge_index_fc, edge_weight_fc):
        """
        Args:
            x: [N, node_feature_dim] 节点特征
            edge_index_sc: [2, E_sc] SC 边索引
            edge_weight_sc: [E_sc] SC 边权重
            edge_index_fc: [2, E_fc] FC 边索引
            edge_weight_fc: [E_fc] FC 边权重
        Returns:
            [N, hidden_dim] 融合后的节点嵌入
        """
        h = x
        
        for i in range(self.num_layers):
            # 在两个图上同时进行消息传递
            h_sc = self.sc_convs[i](h, edge_index_sc, edge_weight_sc)
            h_fc = self.fc_convs[i](h, edge_index_fc, edge_weight_fc)
            
            # 求和融合
            h = h_sc + h_fc  # [N, hidden_dim]
            
            # 激活和正则化
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout(h)
        
        return h


# ==================== 融合方法 C: 结构连接作为骨干 + 功能连接调制 ====================

class FusionMethodC(nn.Module):
    """
    方法 C: SC 作为稳定骨干 + FC 作为调制
    
    流程:
    构建混合邻接矩阵: A_mix = α * A_sc + (1-α) * A_fc
    在混合图上应用 GNN
    """
    def __init__(self, node_feature_dim, hidden_dim, num_layers=2, init_alpha=0.5):
        super(FusionMethodC, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 可学习的融合权重 α
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        
        # GNN 层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch Normalization
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index_sc, edge_weight_sc, edge_index_fc, edge_weight_fc):
        """
        Args:
            x: [N, node_feature_dim] 节点特征
            edge_index_sc: [2, E_sc] SC 边索引
            edge_weight_sc: [E_sc] SC 边权重
            edge_index_fc: [2, E_fc] FC 边索引
            edge_weight_fc: [E_fc] FC 边权重
        Returns:
            [N, hidden_dim] 融合后的节点嵌入
        """
        # 将 alpha 限制在 [0, 1] 范围内
        alpha_clamped = torch.sigmoid(self.alpha)
        
        # 混合边索引和权重
        # 注意：这里假设 SC 和 FC 有相同的边连接模式
        # 如果不同，需要更复杂的合并策略
        edge_index_mix = edge_index_sc  # 使用 SC 的边结构
        edge_weight_mix = alpha_clamped * edge_weight_sc + (1 - alpha_clamped) * edge_weight_fc
        
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_mix, edge_weight_mix)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout(h)
        
        return h


# ==================== 完整模型 ====================

class BrainNetworkFusionModel(nn.Module):
    """
    完整的脑网络融合模型
    整合时间编码器、融合方法、池化和预测器
    """
    def __init__(self, 
                 num_nodes,
                 node_feature_dim=64,
                 hidden_dim=128,
                 num_classes=2,
                 fusion_method='A',
                 num_gnn_layers=2,
                 pooling_type='mean',
                 task_type='classification'):
        """
        Args:
            num_nodes: 脑区数量 (ROI 数量)
            node_feature_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            num_classes: 分类任务的类别数
            fusion_method: 'A', 'B', 或 'C'
            num_gnn_layers: GNN 层数
            pooling_type: 'mean', 'sum', 或 'attention'
            task_type: 'classification' 或 'regression'
        """
        super(BrainNetworkFusionModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.fusion_method_name = fusion_method
        self.task_type = task_type
        
        # 时间编码器（这里用简单的节点特征初始化）
        self.temporal_encoder = TemporalEncoder(
            input_dim=num_nodes,  # 使用邻接矩阵的行作为初始特征
            hidden_dim=hidden_dim,
            output_dim=node_feature_dim
        )
        
        # 选择融合方法
        if fusion_method == 'A':
            self.fusion_module = FusionMethodA(node_feature_dim, hidden_dim, num_gnn_layers)
        elif fusion_method == 'B':
            self.fusion_module = FusionMethodB(node_feature_dim, hidden_dim, num_gnn_layers)
        elif fusion_method == 'C':
            self.fusion_module = FusionMethodC(node_feature_dim, hidden_dim, num_gnn_layers)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # 图池化
        self.pooling = GraphPooling(hidden_dim, pooling_type)
        
        # 预测器
        self.predictor = Predictor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            task_type=task_type
        )
        
    def forward(self, data):
        """
        Args:
            data: PyG Data 对象，包含:
                - x: [N, num_nodes] 节点初始特征（使用 FC 矩阵的行）
                - edge_index_sc: [2, E_sc] SC 边索引
                - edge_weight_sc: [E_sc] SC 边权重
                - edge_index_fc: [2, E_fc] FC 边索引
                - edge_weight_fc: [E_fc] FC 边权重
                - batch: [N] 批次索引
        Returns:
            预测输出: [batch_size, num_classes]
        """
        # 1. 时间编码：生成节点嵌入
        h = self.temporal_encoder(data.x)  # [N, node_feature_dim]
        
        # 2. 图编码（融合）
        h = self.fusion_module(
            h, 
            data.edge_index_sc, 
            data.edge_weight_sc,
            data.edge_index_fc, 
            data.edge_weight_fc
        )  # [N, hidden_dim]
        
        # 3. 图级别池化
        graph_embedding = self.pooling(h, data.batch)  # [batch_size, hidden_dim]
        
        # 4. 预测
        output = self.predictor(graph_embedding)  # [batch_size, num_classes]
        
        return output
    
    def get_fusion_params(self):
        """返回融合方法特定的参数（例如 Method C 的 alpha）"""
        if self.fusion_method_name == 'C':
            alpha = torch.sigmoid(self.fusion_module.alpha).item()
            return {'alpha': alpha}
        return {}
