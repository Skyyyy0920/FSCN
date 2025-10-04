import pickle
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter


class BrainDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        fc = torch.FloatTensor(item['FC'])  # [N, N]
        sc = torch.FloatTensor(item['SC'])  # [N, N]
        label = torch.tensor(item['label'], dtype=torch.long)  # 0/1
        return fc, sc, label


def load_raw_data(pickle_path):
    """
    从pickle文件加载原始数据
    
    返回:
        data_dict: 原始数据字典
        num_nodes: 节点数量
    """
    print(f"正在加载数据: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    # 获取节点数量
    first_key = list(data_dict.keys())[0]
    num_nodes = data_dict[first_key]['FC'].shape[0]
    
    print(f"总样本数: {len(data_dict)}")
    print(f"节点数量: {num_nodes}")
    
    return data_dict, num_nodes


def analyze_tasks(data_dict, min_ratio=0.05):
    """
    分析所有任务的样本分布
    
    注意：-1表示无标签数据，在分析任务有效性时会被忽略，但数据不会被删除
    
    参数:
        data_dict: 数据字典
        min_ratio: 最小样本比例阈值（默认5%）
    
    返回:
        valid_tasks: 有效任务列表
        task_stats: 任务统计信息
    """
    print("\n" + "="*80)
    print("数据集任务分析")
    print("="*80)
    print("注意：标签-1表示无标签数据，在分析时会被忽略（但数据保留用于后续对比学习）")
    
    # 收集所有样本的标签
    all_labels = []
    for idx in data_dict:
        labels = data_dict[idx]['label']
        # 确保label是列表格式
        if not isinstance(labels, list):
            labels = [labels]
        all_labels.append(labels)
    
    # 检测任务数量（假设每个样本的label列表长度相同）
    num_tasks = len(all_labels[0])
    print(f"\n任务总数: {num_tasks}")
    
    task_stats = {}
    valid_tasks = []
    
    for task_idx in range(num_tasks):
        # 提取该任务的所有标签
        task_labels = [labels[task_idx] for labels in all_labels]
        
        # 统计完整类别分布（包括-1）
        counter_all = Counter(task_labels)
        total_all = len(task_labels)
        
        # 只统计有效标签（0和1）的分布
        task_labels_valid = [label for label in task_labels if label != -1]
        counter_valid = Counter(task_labels_valid)
        total_valid = len(task_labels_valid)
        
        # 计算所有类别的比例（用于展示）
        class_ratios_all = {cls: count/total_all for cls, count in counter_all.items()}
        
        # 只基于有效标签（0和1）计算比例和判断是否平衡
        if total_valid > 0:
            class_ratios_valid = {cls: count/total_valid for cls, count in counter_valid.items()}
            min_class_ratio_valid = min(class_ratios_valid.values()) if counter_valid else 0
            is_valid = min_class_ratio_valid >= min_ratio and len(counter_valid) >= 2  # 至少有两个类别
        else:
            class_ratios_valid = {}
            min_class_ratio_valid = 0
            is_valid = False
        
        task_stats[task_idx] = {
            'total': total_all,
            'total_valid': total_valid,
            'distribution': dict(counter_all),
            'distribution_valid': dict(counter_valid),
            'ratios': class_ratios_all,
            'ratios_valid': class_ratios_valid,
            'min_ratio': min_class_ratio_valid,
            'is_valid': is_valid
        }
        
        if is_valid:
            valid_tasks.append(task_idx)
        
        # 打印统计信息
        status = "✓ 有效" if is_valid else "✗ 无效（样本不均衡）"
        print(f"\n任务 {task_idx}: {status}")
        print(f"  总样本数: {total_all} (有标签: {total_valid}, 无标签: {counter_all.get(-1, 0)})")
        print(f"  完整类别分布: {dict(counter_all)}")
        print(f"  有效类别分布 (0,1): {dict(counter_valid)}")
        if total_valid > 0:
            print(f"  有效类别比例: {', '.join([f'类别{k}: {v*100:.2f}%' for k, v in class_ratios_valid.items()])}")
            print(f"  最小有效类别比例: {min_class_ratio_valid*100:.2f}%")
        else:
            print(f"  警告：没有有效标签样本！")
    
    print("\n" + "="*80)
    print(f"有效任务数量: {len(valid_tasks)}/{num_tasks}")
    print(f"有效任务列表: {valid_tasks}")
    print("="*80 + "\n")
    
    return valid_tasks, task_stats


def prepare_task_data(data_dict, task_idx, val_split=0.2, random_state=42):
    # Only use the specific task label
    data_list = []
    for idx in sorted(data_dict.keys()):
        item = data_dict[idx].copy()
        # 提取特定任务的标签
        all_labels = item['label']
        if not isinstance(all_labels, list):
            all_labels = [all_labels]
        
        item['label'] = all_labels[task_idx]
        data_list.append(item)

    labels = [d['label'] for d in data_list]

    train_data, val_data = train_test_split(
        data_list,
        test_size=val_split,
        random_state=random_state,
        stratify=labels
    )
    
    print(f"\n任务 {task_idx} 数据划分:")
    print(f"  训练集: {len(train_data)} 样本")
    train_counter = Counter([d['label'] for d in train_data])
    print(f"    类别分布: {dict(train_counter)}")
    
    print(f"  验证集: {len(val_data)} 样本")
    val_counter = Counter([d['label'] for d in val_data])
    print(f"    类别分布: {dict(val_counter)}")
    
    return train_data, val_data


def get_num_classes(data_list):
    """获取数据集的类别数量（不包括-1）"""
    labels = [d['label'] for d in data_list]
    unique_labels = set(labels)
    # 移除-1（无标签类别）
    unique_labels.discard(-1)
    return len(unique_labels)


def print_task_summary(task_idx, train_data, val_data):
    """打印任务数据摘要"""
    train_labels = [d['label'] for d in train_data]
    val_labels = [d['label'] for d in val_data]
    
    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    
    # 统计有效标签（0和1）
    train_labels_valid = [l for l in train_labels if l != -1]
    val_labels_valid = [l for l in val_labels if l != -1]
    train_counter_valid = Counter(train_labels_valid)
    val_counter_valid = Counter(val_labels_valid)
    
    print(f"\n{'='*60}")
    print(f"任务 {task_idx} 数据摘要")
    print(f"{'='*60}")
    print(f"训练集: {len(train_data)} 样本")
    print(f"  完整分布: {dict(train_counter)}")
    print(f"  有效标签(0,1): {dict(train_counter_valid)} ({len(train_labels_valid)}样本)")
    print(f"验证集: {len(val_data)} 样本")
    print(f"  完整分布: {dict(val_counter)}")
    print(f"  有效标签(0,1): {dict(val_counter_valid)} ({len(val_labels_valid)}样本)")
    print(f"{'='*60}\n")

