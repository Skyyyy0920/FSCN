"""
X-Learner 风格的双模态融合模型

核心思想：
1. Stage 1: 分别训练 FC-only 和 SC-only 基础学习器
2. Stage 2: 学习个体化的模态效应 (modal effect)
3. Stage 3: 使用元学习器融合两者的预测

参考: Künzel et al. "Metalearners for estimating heterogeneous treatment effects using machine learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TransformerEncoder(nn.Module):
    """Transformer Encoder (无 positional embedding)"""
    
    def __init__(self, input_dim, d_model, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        
        # 先投影到 d_model，确保能被 nhead 整除
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 使用 d_model 作为 transformer 的维度（确保能被 nhead 整除）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        """
        x: [B, N, input_dim]
        返回: [B, N, d_model]
        """
        # 先投影到 d_model
        x = self.input_proj(x)  # [B, N, d_model]
        # 通过 transformer
        x = self.transformer(x)  # [B, N, d_model]
        return x


class GCNEncoder(nn.Module):
    """GCN Encoder for SC branch"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        x: [N, input_dim]
        返回: [N, output_dim]
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.norm(x)
        return x


class FCPredictor(nn.Module):
    """
    FC-only 预测器 (Transformer Encoder + Classifier)
    将编码器和分类器集成在一起
    """
    
    def __init__(self, num_nodes, d_model, num_classes, nhead=4, num_layers=2, dropout=0.4):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        # Transformer Encoder
        self.encoder = TransformerEncoder(
            input_dim=num_nodes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, fc_matrix, edge_info=None):
        """
        fc_matrix: [B, N, N]
        返回:
            logits: [B, num_classes]
            features: [B, d_model] 提取的特征
        """
        # 编码
        z = self.encoder(fc_matrix)  # [B, N, d_model]
        
        # 池化
        z_pooled = z.mean(dim=1)  # [B, d_model]
        
        # 分类
        logits = self.classifier(z_pooled)  # [B, num_classes]
        
        return logits, z_pooled


class SCPredictor(nn.Module):
    """
    SC-only 预测器 (GCN Encoder + Classifier)
    将编码器和分类器集成在一起
    """
    
    def __init__(self, num_nodes, d_model, num_classes, dropout=0.4):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        # GCN Encoder
        self.encoder = GCNEncoder(
            input_dim=num_nodes,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=dropout
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, sc_matrix, edge_info=None):
        """
        sc_matrix: [B, N, N]
        返回:
            logits: [B, num_classes]
            features: [B, d_model] 提取的特征
        """
        B = sc_matrix.shape[0]
        
        # 批量处理 GCN
        z_list = []
        for i in range(B):
            sc = sc_matrix[i]
            edge_index, edge_weight = self._sc_to_edge(sc)
            node_features = sc  # [N, N]
            
            z = self.encoder(node_features, edge_index, edge_weight)  # [N, d_model]
            z_list.append(z)
        
        z = torch.stack(z_list, dim=0)  # [B, N, d_model]
        
        # 池化
        z_pooled = z.mean(dim=1)  # [B, d_model]
        
        # 分类
        logits = self.classifier(z_pooled)  # [B, num_classes]
        
        return logits, z_pooled
    
    def _sc_to_edge(self, sc_matrix):
        """将 SC 矩阵转换为边索引和边权重"""
        edge_index = sc_matrix.nonzero(as_tuple=False).t()
        edge_weight = sc_matrix[edge_index[0], edge_index[1]]
        
        if edge_index.shape[1] == 0:
            N = sc_matrix.shape[0]
            edge_index = torch.arange(N, device=sc_matrix.device).repeat(2, 1)
            edge_weight = torch.ones(N, device=sc_matrix.device)
        
        return edge_index, edge_weight


class EffectLearner(nn.Module):
    """
    学习模态效应 (Modal Effect)
    
    输入: 
        - 另一模态的预测概率
        - 当前模态的特征
    输出:
        - 对当前样本，预测两个模态的差异/互补性
    """
    
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        
        self.effect_net = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, features, other_probs):
        """
        features: [B, d_model] 当前模态的特征
        other_probs: [B, num_classes] 另一模态的预测概率
        返回: [B, num_classes] 效应预测
        """
        x = torch.cat([features, other_probs], dim=-1)
        effect = self.effect_net(x)
        return effect


class PropensityNet(nn.Module):
    """
    倾向性网络 (Propensity Network)
    
    学习对于每个样本，应该更信任哪个模态的预测
    输出一个权重，表示 FC 和 SC 的相对重要性
    """
    
    def __init__(self, feature_dim):
        super().__init__()
        
        self.prop_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出 [0, 1] 的权重
        )
    
    def forward(self, fc_features, sc_features):
        """
        fc_features: [B, d_model]
        sc_features: [B, d_model]
        返回: [B, 1] 权重，接近 1 表示更信任 FC，接近 0 表示更信任 SC
        """
        x = torch.cat([fc_features, sc_features], dim=-1)
        weight = self.prop_net(x)
        return weight


class XLearnerModel(nn.Module):
    """
    X-Learner 风格的双模态融合模型
    
    训练分为三个阶段：
    Stage 1: 预训练 FC 和 SC 基础预测器
    Stage 2: 训练效应学习器 (学习模态间的差异)
    Stage 3: 训练倾向性网络 (学习如何融合)
    """
    
    def __init__(self, num_nodes, d_model=128, num_classes=2, dropout=0.4):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Stage 1: 基础预测器（编码器和分类器集成）
        self.fc_predictor = FCPredictor(
            num_nodes=num_nodes,
            d_model=d_model,
            num_classes=num_classes,
            nhead=4,
            num_layers=2,
            dropout=dropout
        )
        
        self.sc_predictor = SCPredictor(
            num_nodes=num_nodes,
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Stage 2: 效应学习器
        # τ_FC: 学习 "如果用 SC 而不是 FC" 的效应
        self.effect_fc2sc = EffectLearner(d_model, num_classes)
        # τ_SC: 学习 "如果用 FC 而不是 SC" 的效应
        self.effect_sc2fc = EffectLearner(d_model, num_classes)
        
        # Stage 3: 倾向性网络
        self.propensity_net = PropensityNet(d_model)
        
        # 训练阶段标志
        self.training_stage = 1  # 1, 2, 或 3
    
    def forward(self, fc_matrix, sc_matrix):
        """
        fc_matrix: [B, N, N]
        sc_matrix: [B, N, N]
        """
        # Stage 1: 基础预测
        fc_logits, fc_features = self.fc_predictor(fc_matrix)
        sc_logits, sc_features = self.sc_predictor(sc_matrix)
        
        fc_probs = F.softmax(fc_logits, dim=-1)
        sc_probs = F.softmax(sc_logits, dim=-1)
        
        if self.training_stage == 1:
            # 仅返回基础预测器的输出
            return {
                'fc_logits': fc_logits,
                'sc_logits': sc_logits,
                'fc_features': fc_features,
                'sc_features': sc_features
            }
        
        # Stage 2: 效应学习
        # 计算个体化的模态效应
        effect_fc = self.effect_fc2sc(fc_features, sc_probs)  # FC 条件下的效应
        effect_sc = self.effect_sc2fc(sc_features, fc_probs)  # SC 条件下的效应
        
        if self.training_stage == 2:
            return {
                'fc_logits': fc_logits,
                'sc_logits': sc_logits,
                'fc_features': fc_features,
                'sc_features': sc_features,
                'effect_fc': effect_fc,
                'effect_sc': effect_sc
            }
        
        # Stage 3: 倾向性加权融合
        # 计算倾向性权重
        prop_weight = self.propensity_net(fc_features, sc_features)  # [B, 1]
        
        # X-Learner 融合公式:
        # τ̂(x) = g(x) * τ̂_0(x) + (1 - g(x)) * τ̂_1(x)
        # 其中 g(x) 是倾向性得分
        
        # 这里我们使用改进的融合策略：
        # 1. 基于效应的预测调整
        fc_adjusted = fc_logits + effect_sc  # FC 预测 + 来自 SC 的效应
        sc_adjusted = sc_logits + effect_fc  # SC 预测 + 来自 FC 的效应
        
        # 2. 倾向性加权融合
        final_logits = prop_weight * fc_adjusted + (1 - prop_weight) * sc_adjusted
        
        return {
            'final_logits': final_logits,
            'fc_logits': fc_logits,
            'sc_logits': sc_logits,
            'fc_adjusted': fc_adjusted,
            'sc_adjusted': sc_adjusted,
            'prop_weight': prop_weight,
            'effect_fc': effect_fc,
            'effect_sc': effect_sc
        }
    
    def set_training_stage(self, stage):
        """设置训练阶段"""
        assert stage in [1, 2, 3], "Stage must be 1, 2, or 3"
        self.training_stage = stage
        print(f"\n{'='*60}")
        print(f"X-Learner 训练阶段切换: Stage {stage}")
        if stage == 1:
            print("Stage 1: 训练基础预测器 (FC-only & SC-only)")
        elif stage == 2:
            print("Stage 2: 训练效应学习器 (学习模态间差异)")
        else:
            print("Stage 3: 训练倾向性网络 (学习融合权重)")
        print(f"{'='*60}\n")
    
    def freeze_stage(self, stage):
        """冻结指定阶段的参数"""
        if stage == 1:
            # 冻结基础预测器
            for param in self.fc_predictor.parameters():
                param.requires_grad = False
            for param in self.sc_predictor.parameters():
                param.requires_grad = False
            print("✓ Frozen Stage 1 (Base Predictors)")
        
        elif stage == 2:
            # 冻结效应学习器
            for param in self.effect_fc2sc.parameters():
                param.requires_grad = False
            for param in self.effect_sc2fc.parameters():
                param.requires_grad = False
            print("✓ Frozen Stage 2 (Effect Learners)")
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ Unfrozen all parameters")


def test_xlearner_model():
    """测试 X-Learner 模型"""
    print("\n" + "="*60)
    print("Testing X-Learner Model")
    print("="*60 + "\n")
    
    # 模拟数据
    batch_size = 4
    num_nodes = 200
    num_classes = 2
    
    fc_matrix = torch.randn(batch_size, num_nodes, num_nodes)
    sc_matrix = torch.abs(torch.randn(batch_size, num_nodes, num_nodes))
    
    # 创建模型
    model = XLearnerModel(
        num_nodes=num_nodes,
        d_model=128,
        num_classes=num_classes
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试 Stage 1
    print("\n--- Testing Stage 1 ---")
    model.set_training_stage(1)
    out1 = model(fc_matrix, sc_matrix)
    print(f"FC logits shape: {out1['fc_logits'].shape}")
    print(f"SC logits shape: {out1['sc_logits'].shape}")
    
    # 测试 Stage 2
    print("\n--- Testing Stage 2 ---")
    model.set_training_stage(2)
    out2 = model(fc_matrix, sc_matrix)
    print(f"Effect FC shape: {out2['effect_fc'].shape}")
    print(f"Effect SC shape: {out2['effect_sc'].shape}")
    
    # 测试 Stage 3
    print("\n--- Testing Stage 3 ---")
    model.set_training_stage(3)
    out3 = model(fc_matrix, sc_matrix)
    print(f"Final logits shape: {out3['final_logits'].shape}")
    print(f"Propensity weight shape: {out3['prop_weight'].shape}")
    print(f"Sample propensity weights: {out3['prop_weight'][:3, 0]}")
    
    print("\n" + "="*60)
    print("✓ X-Learner Model Test Passed!")
    print("="*60 + "\n")


if __name__ == '__main__':
    test_xlearner_model()

