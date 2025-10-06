#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import to_undirected, add_self_loops
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class BrainNetworkDataset(Dataset):
    """
    脑网络数据集类，支持NeuroGraphDataset的五个数据集
    """
    
    def __init__(self, 
                 root: str = './data',
                 name: str = 'HCPGender',
                 transform=None,
                 pre_transform=None,
                 threshold: float = 0.1):
        """
        Args:
            root: 数据根目录
            name: NeuroGraphDataset名称 ('HCPGender', 'HCPTask', 'HCPAge', 'HCPFI', 'HCPWM')
            transform: 数据变换
            pre_transform: 预处理变换
            threshold: 连接阈值，用于构建邻接矩阵
        """
        self.name = name
        self.threshold = threshold
        self.data_list = []
        
        # 验证数据集名称
        valid_names = ['HCPGender', 'HCPTask', 'HCPAge', 'HCPFI', 'HCPWM']
        if name not in valid_names:
            raise ValueError(f"不支持的数据集名称: {name}。支持的数据集: {valid_names}")
        
        super().__init__(root, transform, pre_transform)
        
        self._load_neurograph_dataset()
    
    def _load_neurograph_dataset(self):
        """加载NeuroGraphDataset数据集"""
        try:
            from torch_geometric.datasets import NeuroGraphDataset
            
            print(f"正在加载NeuroGraphDataset: {self.name}")
            
            # 加载指定的NeuroGraphDataset
            dataset = NeuroGraphDataset(root=self.root, name=self.name)
            
            print(f"成功加载 {self.name} 数据集，包含 {len(dataset)} 个样本")
            
            # 转换为我们的格式
            for i, data in enumerate(dataset):
                # 获取节点特征
                if hasattr(data, 'x') and data.x is not None:
                    x = data.x
                else:
                    # 如果没有节点特征，使用单位矩阵
                    x = torch.eye(data.num_nodes)
                
                # 获取边索引
                if hasattr(data, 'edge_index') and data.edge_index is not None:
                    edge_index = data.edge_index
                else:
                    # 如果有邻接矩阵，从中构建边索引
                    if hasattr(data, 'adj') and data.adj is not None:
                        edge_index = self._build_edge_index_from_adj(data.adj)
                    else:
                        # 创建空的边索引
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                
                # 获取标签
                if hasattr(data, 'y') and data.y is not None:
                    y = data.y
                else:
                    # 如果没有标签，创建默认标签
                    y = torch.tensor([0], dtype=torch.long)
                
                # 确保标签是正确的格式
                if y.dim() == 0:
                    y = y.unsqueeze(0)
                
                # 创建Data对象
                graph_data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    num_nodes=data.num_nodes if hasattr(data, 'num_nodes') else x.size(0)
                )
                
                # 如果原始数据有其他属性，也保存下来
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    graph_data.edge_attr = data.edge_attr
                
                self.data_list.append(graph_data)
            
            print(f"数据转换完成，节点特征维度: {self.data_list[0].x.shape[1]}")
            
        except ImportError as e:
            print(f"警告: 无法导入NeuroGraphDataset: {e}")
            print("将使用模拟数据进行测试")
            self._generate_synthetic_data()
        except Exception as e:
            print(f"加载NeuroGraphDataset时出错: {e}")
            print("将使用模拟数据进行测试")
            self._generate_synthetic_data()
    
    
    def _build_edge_index_from_adj(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """从邻接张量构建边索引"""
        edge_indices = torch.nonzero(adj_matrix, as_tuple=False).t()
        edge_index = add_self_loops(edge_indices)[0]
        edge_index = to_undirected(edge_index)
        return edge_index
    
    def _generate_synthetic_data(self, num_samples: int = 1000, num_nodes: int = 100):
        """生成合成数据用于测试"""
        print(f"生成 {num_samples} 个合成脑网络样本...")
        
        for i in range(num_samples):
            # 生成随机FC矩阵
            fc_matrix = np.random.rand(num_nodes, num_nodes)
            fc_matrix = (fc_matrix + fc_matrix.T) / 2  # 对称化
            np.fill_diagonal(fc_matrix, 1.0)  # 对角线为1
            
            # 生成随机SC矩阵
            sc_matrix = np.random.rand(num_nodes, num_nodes)
            sc_matrix = (sc_matrix + sc_matrix.T) / 2  # 对称化
            np.fill_diagonal(sc_matrix, 1.0)  # 对角线为1
            
            # 提取节点特征
            node_features = self._extract_node_features(fc_matrix, sc_matrix)
            
            # 构建边索引
            edge_index = self._build_edge_index_from_matrix(sc_matrix)
            
            # 随机标签
            y = torch.randint(0, 2, (1,))
            
            # 创建Data对象
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                y=y,
                fc_matrix=torch.tensor(fc_matrix, dtype=torch.float32),
                sc_matrix=torch.tensor(sc_matrix, dtype=torch.float32),
                num_nodes=num_nodes
            )
            
            self.data_list.append(graph_data)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def get_data_splits(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """获取训练/验证/测试数据分割"""
        indices = list(range(len(self.data_list)))
        
        # 首先分割出测试集
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=[data.y.item() for data in self.data_list]
        )
        
        # 再从训练集中分割出验证集
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size/(1-test_size), random_state=random_state, 
            stratify=[self.data_list[i].y.item() for i in train_val_indices]
        )
        
        return train_indices, val_indices, test_indices


def create_data_loaders(dataset: BrainNetworkDataset, 
                       batch_size: int = 32,
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42):
    """创建数据加载器"""
    from torch_geometric.loader import DataLoader
    
    # 获取数据分割
    train_indices, val_indices, test_indices = dataset.get_data_splits(test_size, val_size, random_state)
    
    # 创建子集
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 计算类别数
    all_labels = torch.cat([data.y for data in dataset])
    if dataset.name in ['HCPFI', 'HCPWM']:  # 回归任务
        num_classes = 1
    else:  # 分类任务
        num_classes = len(torch.unique(all_labels))
    
    return train_loader, val_loader, test_loader, {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'num_features': dataset[0].x.shape[1],
        'num_classes': num_classes,
        'task_type': 'regression' if dataset.name in ['HCPFI', 'HCPWM'] else 'classification'
    }


if __name__ == "__main__":
    # 测试数据加载器
    print("测试NeuroGraphDataset数据加载器...")
    
    # 测试所有五个数据集
    dataset_names = ['HCPGender', 'HCPTask', 'HCPAge', 'HCPFI', 'HCPWM']
    
    for i, dataset_name in enumerate(dataset_names, 1):
        print(f"\n{i}. 测试 {dataset_name} 数据集:")
        try:
            dataset = BrainNetworkDataset(name=dataset_name)
            print(f"数据集样本数: {len(dataset)}")
            
            if len(dataset) > 0:
                print(f"节点特征维度: {dataset[0].x.shape}")
                print(f"边数量: {dataset[0].edge_index.shape[1]}")
                print(f"标签: {dataset[0].y}")
                print(f"标签形状: {dataset[0].y.shape}")
                
                # 测试数据加载器
                train_loader, val_loader, test_loader, info = create_data_loaders(dataset)
                print(f"训练集大小: {info['train_size']}")
                print(f"验证集大小: {info['val_size']}")
                print(f"测试集大小: {info['test_size']}")
                print(f"特征维度: {info['num_features']}")
                print(f"类别数/输出维度: {info['num_classes']}")
                print(f"任务类型: {info['task_type']}")
                
                # 测试一个批次
                for batch in train_loader:
                    print(f"批次大小: {batch.batch_size}")
                    print(f"节点特征形状: {batch.x.shape}")
                    print(f"边索引形状: {batch.edge_index.shape}")
                    print(f"标签形状: {batch.y.shape}")
                    break
                    
        except Exception as e:
            print(f"❌ 测试 {dataset_name} 失败: {e}")
    
    print("\n✅ 数据加载器测试完成!")

