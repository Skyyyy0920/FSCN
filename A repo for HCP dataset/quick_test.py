#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速测试GraphVAE和GraphDiffusion模型修复
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from models import GraphVAE, GraphDiffusion

def test_graphvae_simple():
    """简单测试GraphVAE"""
    print("测试GraphVAE...")
    
    # 创建简单测试数据
    batch_size = 2
    num_nodes_per_graph = 10
    input_dim = 8
    
    # 节点特征
    x = torch.randn(batch_size * num_nodes_per_graph, input_dim)
    
    # 边索引（简单的链式结构）
    edge_list = []
    for b in range(batch_size):
        offset = b * num_nodes_per_graph
        for i in range(num_nodes_per_graph - 1):
            edge_list.append([offset + i, offset + i + 1])
            edge_list.append([offset + i + 1, offset + i])  # 无向边
    
    edge_index = torch.tensor(edge_list).t()
    
    # 批次索引
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes_per_graph)
    
    # 标签
    y = torch.randint(0, 2, (batch_size,))
    
    try:
        # 创建模型
        model = GraphVAE(
            input_dim=input_dim,
            hidden_dims=[16, 8],
            latent_dim=16,
            num_classes=2,
            dropout=0.5
        )
        
        print(f"GraphVAE参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 前向传播
        output = model(x, edge_index, batch)
        print(f"输出键: {list(output.keys())}")
        print(f"分类输出形状: {output['classification'].shape}")
        print(f"重建输出形状: {output['reconstructed'].shape}")
        
        # 测试损失计算
        graph_features = global_mean_pool(x, batch)
        loss_dict = model.compute_loss(output, y, graph_features)
        print(f"总损失: {loss_dict['total_loss'].item():.4f}")
        print(f"分类损失: {loss_dict['ce_loss'].item():.4f}")
        print(f"重建损失: {loss_dict['recon_loss'].item():.4f}")
        print(f"KL损失: {loss_dict['kl_loss'].item():.4f}")
        
        print("✅ GraphVAE测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ GraphVAE测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graphdiffusion_simple():
    """简单测试GraphDiffusion"""
    print("\n测试GraphDiffusion...")
    
    # 创建简单测试数据
    batch_size = 2
    num_nodes_per_graph = 10
    input_dim = 8
    
    # 节点特征
    x = torch.randn(batch_size * num_nodes_per_graph, input_dim)
    
    # 边索引
    edge_list = []
    for b in range(batch_size):
        offset = b * num_nodes_per_graph
        for i in range(num_nodes_per_graph - 1):
            edge_list.append([offset + i, offset + i + 1])
            edge_list.append([offset + i + 1, offset + i])
    
    edge_index = torch.tensor(edge_list).t()
    
    # 批次索引
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes_per_graph)
    
    try:
        # 创建模型（使用更小的配置）
        model = GraphDiffusion(
            input_dim=input_dim,
            hidden_dims=[16, 8],
            num_classes=2,
            dropout=0.5,
            num_timesteps=100
        )
        
        print(f"GraphDiffusion参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 测试分类模式
        classification_output = model(x, edge_index, batch, mode='classification')
        print(f"分类输出形状: {classification_output.shape}")
        
        # 测试扩散模式
        timestep = torch.randint(0, 100, (batch_size,))
        diffusion_output = model(x, edge_index, batch, mode='diffusion', timestep=timestep)
        print(f"扩散输出形状: {diffusion_output.shape}")
        
        # 测试扩散损失计算
        encoded_features = model.encode(x, edge_index, batch)
        print(f"编码特征形状: {encoded_features.shape}")
        
        loss = model.compute_diffusion_loss(encoded_features, timestep, edge_index, batch)
        print(f"扩散损失: {loss.item():.4f}")
        
        print("✅ GraphDiffusion测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ GraphDiffusion测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始快速测试...")
    
    vae_success = test_graphvae_simple()
    diffusion_success = test_graphdiffusion_simple()
    
    print("\n" + "="*50)
    print("测试结果:")
    print(f"GraphVAE: {'✅ 成功' if vae_success else '❌ 失败'}")
    print(f"GraphDiffusion: {'✅ 成功' if diffusion_success else '❌ 失败'}")
    
    if vae_success and diffusion_success:
        print("\n🎉 所有模型修复成功！")
    else:
        print("\n⚠️ 部分模型仍有问题")

