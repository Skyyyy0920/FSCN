import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


class BrainNetworkDataset(Dataset):
    """
    脑网络数据集
    加载 FC 和 SC 连接矩阵，转换为图数据
    """
    def __init__(self, data_dict, indices=None, threshold=0.1):
        """
        Args:
            data_dict: 包含所有受试者数据的字典
            indices: 使用的受试者索引列表（用于训练/测试分割）
            threshold: 边权重阈值，小于此值的边将被移除
        """
        self.data_dict = data_dict
        self.indices = indices if indices is not None else list(data_dict.keys())
        self.threshold = threshold

        first_key = self.indices[0]
        self.num_nodes = data_dict[first_key]['FC'].shape[0]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        获取一个受试者的图数据
        """
        subject_idx = self.indices[idx]
        subject_data = self.data_dict[subject_idx]
        
        # 获取连接矩阵
        fc = subject_data['FC']  # [N, N]
        sc = subject_data['SC']  # [N, N]
        label = subject_data['label']
        
        # 转换为 PyG Data 对象
        data = self._create_pyg_data(fc, sc, label)
        
        return data
    
    def _create_pyg_data(self, fc, sc, label):
        """
        将连接矩阵转换为 PyTorch Geometric Data 对象
        
        Args:
            fc: [N, N] 功能连接矩阵
            sc: [N, N] 结构连接矩阵
            label: 标签（单个值或列表）
        
        Returns:
            PyG Data 对象
        """
        num_nodes = fc.shape[0]
        
        # 1. 节点特征：使用 FC 矩阵的行作为初始特征
        x = torch.FloatTensor(fc)  # [N, N]
        
        # 2. 创建统一的边结构（SC 和 FC 的并集）
        edge_index_unified, edge_weight_sc, edge_weight_fc = self._create_unified_edges(fc, sc, self.threshold)
        
        # 3. 处理标签
        if isinstance(label, (list, np.ndarray)):
            # 对于多标签情况，我们使用第一个非负标签作为主要标签
            if len(label) > 1:
                # 找到第一个非负标签
                main_label = None
                for l in label:
                    if l >= 0:  # 假设 -1 表示缺失值
                        main_label = l
                        break
                if main_label is None:
                    main_label = 0  # 默认标签
                y = torch.LongTensor([main_label])
            else:
                y = torch.LongTensor([label[0]])
        else:
            y = torch.LongTensor([label])
        
        # 创建 Data 对象（使用统一的边索引）
        data = Data(
            x=x,
            edge_index_sc=edge_index_unified,
            edge_weight_sc=edge_weight_sc,
            edge_index_fc=edge_index_unified,
            edge_weight_fc=edge_weight_fc,
            y=y,
            num_nodes=num_nodes
        )
        
        return data
    
    def _adj_to_edge(self, adj_matrix, threshold=0.0):
        """
        将邻接矩阵转换为边索引和边权重
        
        Args:
            adj_matrix: [N, N] 邻接矩阵
            threshold: 边权重阈值
        
        Returns:
            edge_index: [2, E] 边索引
            edge_weight: [E] 边权重
        """
        # 取绝对值并应用阈值
        adj = np.abs(adj_matrix)
        
        # 找到所有大于阈值的边
        row, col = np.where(adj > threshold)
        edge_weight = adj[row, col]
        
        # 转换为 PyTorch 张量
        edge_index = torch.LongTensor(np.stack([row, col], axis=0))  # [2, E]
        edge_weight = torch.FloatTensor(edge_weight)  # [E]
        
        return edge_index, edge_weight
    
    def _create_unified_edges(self, fc, sc, threshold=0.0):
        """
        创建统一的边结构，包含 SC 和 FC 的所有边
        
        Args:
            fc: [N, N] 功能连接矩阵
            sc: [N, N] 结构连接矩阵
            threshold: 边权重阈值
        
        Returns:
            edge_index: [2, E] 统一的边索引
            edge_weight_sc: [E] SC 边权重
            edge_weight_fc: [E] FC 边权重
        """
        # 处理 FC 和 SC
        fc_abs = np.abs(fc)
        sc_abs = np.abs(sc)
        
        # 找出两个图中所有超过阈值的边（并集）
        fc_mask = fc_abs > threshold
        sc_mask = sc_abs > threshold
        combined_mask = fc_mask | sc_mask
        
        # 获取所有边的索引
        row, col = np.where(combined_mask)
        
        # 为每条边分配权重
        edge_weight_fc = fc_abs[row, col]
        edge_weight_sc = sc_abs[row, col]
        
        # 转换为 PyTorch 张量
        edge_index = torch.LongTensor(np.stack([row, col], axis=0))  # [2, E]
        edge_weight_fc = torch.FloatTensor(edge_weight_fc)  # [E]
        edge_weight_sc = torch.FloatTensor(edge_weight_sc)  # [E]
        
        return edge_index, edge_weight_sc, edge_weight_fc


def load_data_dict(file_path):
    """
    从 pickle 文件加载数据字典
    
    Args:
        file_path: pickle 文件路径
    
    Returns:
        data_dict: 数据字典
    """
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    return data_dict


def create_data_splits(data_dict, test_size=0.2, val_size=0.1, random_state=42):
    """
    创建训练集、验证集和测试集的索引分割
    
    Args:
        data_dict: 数据字典
        test_size: 测试集比例
        val_size: 验证集比例（相对于训练集）
        random_state: 随机种子
    
    Returns:
        train_indices, val_indices, test_indices
    """
    all_indices = list(data_dict.keys())
    
    # 提取所有标签用于分层采样
    labels = []
    for idx in all_indices:
        label = data_dict[idx]['label']
        if isinstance(label, list):
            # 对于多标签情况，使用第一个非负标签
            main_label = None
            for l in label:
                if l >= 0:  # 假设 -1 表示缺失值
                    main_label = l
                    break
            if main_label is None:
                main_label = 0  # 默认标签
            labels.append(main_label)
        else:
            labels.append(label)

    train_val_indices, test_indices = train_test_split(
        all_indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )

    train_labels = []
    for idx in train_val_indices:
        label = data_dict[idx]['label']
        if isinstance(label, list):
            # 对于多标签情况，使用第一个非负标签
            main_label = None
            for l in label:
                if l >= 0:  # 假设 -1 表示缺失值
                    main_label = l
                    break
            if main_label is None:
                main_label = 0  # 默认标签
            train_labels.append(main_label)
        else:
            train_labels.append(label)
    
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size,
        random_state=random_state,
        stratify=train_labels
    )
    
    print(f"Data split:")
    print(f"train: {len(train_indices)} 样本")
    print(f"val: {len(val_indices)} 样本")
    print(f"test: {len(test_indices)} 样本")
    
    return train_indices, val_indices, test_indices


def get_num_classes(data_dict):
    """
    获取分类任务的类别数
    
    Args:
        data_dict: 数据字典
    
    Returns:
        num_classes: 类别数
    """
    all_labels = []
    for idx in data_dict.keys():
        label = data_dict[idx]['label']
        if isinstance(label, list):
            # 对于多标签情况，使用第一个非负标签
            main_label = None
            for l in label:
                if l >= 0:  # 假设 -1 表示缺失值
                    main_label = l
                    break
            if main_label is None:
                main_label = 0  # 默认标签
            all_labels.append(main_label)
        else:
            all_labels.append(label)
    
    num_classes = len(set(all_labels))
    print(f"检测到 {num_classes} 个类别")
    
    return num_classes


def collate_fn(batch):
    """
    自定义 collate 函数，用于 DataLoader
    将一批图数据合并为一个大图（PyG 的标准做法）
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)
