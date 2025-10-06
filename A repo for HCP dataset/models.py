#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np
from typing import Optional, Tuple, List
import math


class GCNClassifier(nn.Module):
    """
    GCN分类器用于脑网络数据预测
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 num_classes: int = 2,
                 dropout: float = 0.5,
                 pool_type: str = 'mean'):
        super(GCNClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.pool_type = pool_type
        
        # GCN层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 第一层
        self.convs.append(GCNConv(input_dim, hidden_dims[0]))
        self.bns.append(nn.BatchNorm1d(hidden_dims[0]))
        
        # 隐藏层
        for i in range(1, len(hidden_dims)):
            self.convs.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
            self.bns.append(nn.BatchNorm1d(hidden_dims[i]))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
        # 池化函数
        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'max':
            self.pool = global_max_pool
        elif pool_type == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"不支持的池化类型: {pool_type}")
    
    def forward(self, x, edge_index, batch):
        # GCN层
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 图级别池化
        x = self.pool(x, batch)
        
        # 分类
        x = self.classifier(x)
        
        return x


class GraphSAGEClassifier(nn.Module):
    """
    GraphSAGE分类器用于脑网络数据预测
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 num_classes: int = 2,
                 dropout: float = 0.5,
                 pool_type: str = 'mean'):
        super(GraphSAGEClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.pool_type = pool_type
        
        # GraphSAGE层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 第一层
        self.convs.append(SAGEConv(input_dim, hidden_dims[0]))
        self.bns.append(nn.BatchNorm1d(hidden_dims[0]))
        
        # 隐藏层
        for i in range(1, len(hidden_dims)):
            self.convs.append(SAGEConv(hidden_dims[i-1], hidden_dims[i]))
            self.bns.append(nn.BatchNorm1d(hidden_dims[i]))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
        # 池化函数
        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'max':
            self.pool = global_max_pool
        elif pool_type == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"不支持的池化类型: {pool_type}")
    
    def forward(self, x, edge_index, batch):
        # GraphSAGE层
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 图级别池化
        x = self.pool(x, batch)
        
        # 分类
        x = self.classifier(x)
        
        return x


class GATClassifier(nn.Module):
    """
    GAT分类器用于脑网络数据预测
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 num_classes: int = 2,
                 dropout: float = 0.5,
                 pool_type: str = 'mean',
                 heads: List[int] = [4, 4]):
        super(GATClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.pool_type = pool_type
        self.heads = heads
        
        # GAT层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 第一层
        self.convs.append(GATConv(input_dim, hidden_dims[0], heads=heads[0], dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_dims[0] * heads[0]))
        
        # 隐藏层
        for i in range(1, len(hidden_dims)):
            self.convs.append(GATConv(hidden_dims[i-1] * heads[i-1], hidden_dims[i], 
                                    heads=heads[i], dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dims[i] * heads[i]))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1] * heads[-1], hidden_dims[-1] * heads[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] * heads[-1] // 2, num_classes)
        )
        
        # 池化函数
        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'max':
            self.pool = global_max_pool
        elif pool_type == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"不支持的池化类型: {pool_type}")
    
    def forward(self, x, edge_index, batch):
        # GAT层
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 图级别池化
        x = self.pool(x, batch)
        
        # 分类
        x = self.classifier(x)
        
        return x


class GraphVAE(nn.Module):
    """
    Graph VAE模型用于脑网络数据预测
    Encoder(G) → z (128维)
    Decoder(z) → Ĝ（辅助重建）
    MLP(z) → 性别分类 (男/女)
    Loss = CE(性别) + λ·MSE(G,Ĝ) + KL(z||N(0,1))
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 latent_dim: int = 128,
                 num_classes: int = 2,
                 dropout: float = 0.5,
                 pool_type: str = 'mean'):
        super(GraphVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.pool_type = pool_type
        
        # 编码器
        self.encoder_convs = nn.ModuleList()
        self.encoder_bns = nn.ModuleList()
        
        # 第一层
        self.encoder_convs.append(GCNConv(input_dim, hidden_dims[0]))
        self.encoder_bns.append(nn.BatchNorm1d(hidden_dims[0]))
        
        # 隐藏层
        for i in range(1, len(hidden_dims)):
            self.encoder_convs.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
            self.encoder_bns.append(nn.BatchNorm1d(hidden_dims[i]))
        
        # 池化函数
        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'max':
            self.pool = global_max_pool
        elif pool_type == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"不支持的池化类型: {pool_type}")
        
        # VAE编码器
        self.mu_encoder = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_encoder = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], input_dim)  # 重建节点特征而不是邻接矩阵
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_classes)
        )
    
    def encode(self, x, edge_index, batch):
        """编码器前向传播"""
        # GCN层
        for i, (conv, bn) in enumerate(zip(self.encoder_convs, self.encoder_bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 图级别池化
        x = self.pool(x, batch)
        
        # VAE编码
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """解码器前向传播"""
        return self.decoder(z)
    
    def forward(self, x, edge_index, batch):
        # 编码
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        
        # 解码重建
        reconstructed = self.decode(z)
        
        # 分类
        classification = self.classifier(z)
        
        return {
            'classification': classification,
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def compute_loss(self, output, target, original_features=None, lambda_recon=1.0, lambda_kl=1.0):
        """计算VAE损失"""
        # 分类损失
        ce_loss = F.cross_entropy(output['classification'], target)
        
        # 重建损失 - 简化为重建图级别特征
        # 使用平均池化的节点特征作为目标
        batch_size = output['reconstructed'].size(0)
        
        # 如果没有提供原始特征，使用零重建损失
        if original_features is None:
            recon_loss = torch.tensor(0.0, device=output['reconstructed'].device)
        else:
            # 计算图级别特征的重建损失
            # original_features应该是图级别的特征 (batch_size, feature_dim)
            if original_features.dim() > 2:
                # 如果是节点级别特征，进行平均池化
                original_features = original_features.mean(dim=1)
            
            # 确保维度匹配
            if original_features.size(1) != output['reconstructed'].size(1):
                # 如果维度不匹配，使用线性层调整
                if not hasattr(self, 'feature_adapter'):
                    self.feature_adapter = nn.Linear(original_features.size(1), output['reconstructed'].size(1)).to(output['reconstructed'].device)
                original_features = self.feature_adapter(original_features)
            
            recon_loss = F.mse_loss(output['reconstructed'], original_features)
        
        # KL散度损失 - 归一化到批次大小
        kl_loss = -0.5 * torch.mean(torch.sum(1 + output['logvar'] - output['mu'].pow(2) - output['logvar'].exp(), dim=1))
        
        # 总损失
        total_loss = ce_loss + lambda_recon * recon_loss + lambda_kl * kl_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class GraphDiffusion(nn.Module):
    """
    Graph Diffusion模型用于脑网络数据预测
    预训练：图加噪声 → 扩散模型去噪
    Fine-tune：拿 encoder 的表示 z → MLP 分类
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 num_classes: int = 2,
                 dropout: float = 0.5,
                 pool_type: str = 'mean',
                 num_timesteps: int = 1000):
        super(GraphDiffusion, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.pool_type = pool_type
        self.num_timesteps = num_timesteps
        
        # 编码器
        self.encoder_convs = nn.ModuleList()
        self.encoder_bns = nn.ModuleList()
        
        # 第一层
        self.encoder_convs.append(GCNConv(input_dim, hidden_dims[0]))
        self.encoder_bns.append(nn.BatchNorm1d(hidden_dims[0]))
        
        # 隐藏层
        for i in range(1, len(hidden_dims)):
            self.encoder_convs.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
            self.encoder_bns.append(nn.BatchNorm1d(hidden_dims[i]))
        
        # 池化函数
        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'max':
            self.pool = global_max_pool
        elif pool_type == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"不支持的池化类型: {pool_type}")
        
        # 扩散模型 - 减小输出维度以节省内存
        self.diffusion_net = nn.Sequential(
            nn.Linear(hidden_dims[-1] + 1, hidden_dims[-1]),  # +1 for timestep
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], input_dim)  # 只重建节点特征而不是邻接矩阵
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
        # 时间嵌入
        self.time_embedding = nn.Embedding(num_timesteps, 1)
    
    def encode(self, x, edge_index, batch):
        """编码器前向传播"""
        # GCN层
        for i, (conv, bn) in enumerate(zip(self.encoder_convs, self.encoder_bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 图级别池化
        x = self.pool(x, batch)
        
        return x
    
    def forward_diffusion(self, x, edge_index, batch, timestep=None):
        """扩散模型前向传播"""
        # 编码
        encoded = self.encode(x, edge_index, batch)
        
        if timestep is None:
            timestep = torch.randint(0, self.num_timesteps, (encoded.size(0),), device=encoded.device)
        
        # 时间嵌入
        time_emb = self.time_embedding(timestep)
        
        # 扩散网络
        diffusion_input = torch.cat([encoded, time_emb], dim=1)
        noise_pred = self.diffusion_net(diffusion_input)
        
        return noise_pred
    
    def forward_classification(self, x, edge_index, batch):
        """分类前向传播"""
        # 编码
        encoded = self.encode(x, edge_index, batch)
        
        # 分类
        classification = self.classifier(encoded)
        
        return classification
    
    def forward(self, x, edge_index, batch, mode='classification', timestep=None):
        """前向传播"""
        if mode == 'diffusion':
            return self.forward_diffusion(x, edge_index, batch, timestep)
        elif mode == 'classification':
            return self.forward_classification(x, edge_index, batch)
        else:
            raise ValueError(f"不支持的模式: {mode}")
    
    def add_noise(self, original_features, timestep):
        """添加噪声到特征"""
        noise = torch.randn_like(original_features)
        alpha = 1.0 - timestep.float() / self.num_timesteps
        alpha = alpha.view(-1, 1)  # 调整维度以匹配特征
        noisy_features = torch.sqrt(alpha) * original_features + torch.sqrt(1 - alpha) * noise
        return noisy_features, noise
    
    def compute_diffusion_loss(self, original_features, timestep, edge_index, batch):
        """计算扩散损失"""
        # 添加噪声到图级别特征
        noisy_features, noise = self.add_noise(original_features, timestep)
        
        # 使用带噪声的特征进行前向传播
        # 这里我们需要重新构造输入
        batch_size = noisy_features.size(0)
        num_nodes_per_graph = original_features.size(1) if original_features.dim() > 1 else 1
        
        # 创建虚拟节点特征用于扩散训练
        dummy_x = torch.randn(batch_size * num_nodes_per_graph, self.input_dim, device=original_features.device)
        dummy_batch = torch.repeat_interleave(torch.arange(batch_size, device=original_features.device), num_nodes_per_graph)
        
        # 预测噪声
        noise_pred = self.forward_diffusion(dummy_x, edge_index, dummy_batch, timestep)
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss


def get_model(model_name: str, input_dim: int, num_classes: int = 2, task_type: str = 'classification', **kwargs):
    """获取指定模型"""
    models = {
        'gcn': GCNClassifier,
        'graphsage': GraphSAGEClassifier,
        'gat': GATClassifier,
        'graphvae': GraphVAE,
        'graphdiffusion': GraphDiffusion
    }
    
    if model_name not in models:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 对于回归任务，将num_classes设置为1
    if task_type == 'regression':
        num_classes = 1
    
    return models[model_name](input_dim=input_dim, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # 测试模型
    print("测试模型...")
    
    # 测试参数
    input_dim = 8
    num_classes = 2
    batch_size = 4
    num_nodes = 10
    
    # 创建测试数据
    x = torch.randn(num_nodes * batch_size, input_dim)
    edge_index = torch.randint(0, num_nodes * batch_size, (2, num_nodes * batch_size * 2))
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    y = torch.randint(0, num_classes, (batch_size,))
    
    # 测试各种模型
    models_to_test = ['gcn', 'graphsage', 'gat', 'graphvae', 'graphdiffusion']
    
    for model_name in models_to_test:
        print(f"\n测试 {model_name.upper()} 模型:")
        try:
            model = get_model(model_name, input_dim, num_classes)
            print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
            
            if model_name == 'graphvae':
                output = model(x, edge_index, batch)
                print(f"输出键: {output.keys()}")
                print(f"分类输出形状: {output['classification'].shape}")
            elif model_name == 'graphdiffusion':
                # 测试分类模式
                output = model(x, edge_index, batch, mode='classification')
                print(f"分类输出形状: {output.shape}")
                
                # 测试扩散模式
                timestep = torch.randint(0, 1000, (batch_size,))
                noise_pred = model(x, edge_index, batch, mode='diffusion', timestep=timestep)
                print(f"噪声预测形状: {noise_pred.shape}")
            else:
                output = model(x, edge_index, batch)
                print(f"输出形状: {output.shape}")
            
            print("✅ 模型测试成功")
            
        except Exception as e:
            print(f"❌ 模型测试失败: {e}")

