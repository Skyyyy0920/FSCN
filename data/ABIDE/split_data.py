"""
脚本用于将 ABIDE 数据集按照 7:1:2 的比例分割成训练集、验证集和测试集
"""
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os


def split_abide_data(input_path, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):
    """
    将 ABIDE 数据集分割成训练集、验证集和测试集
    
    Args:
        input_path: 输入的 .npy 文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例 (默认 0.7)
        val_ratio: 验证集比例 (默认 0.1)
        test_ratio: 测试集比例 (默认 0.2)
        random_state: 随机种子
    """
    print("=" * 80)
    print("ABIDE 数据集分割工具")
    print("=" * 80)
    
    # 加载原始数据
    print(f"\n正在加载数据: {input_path}")
    abide_data = np.load(input_path, allow_pickle=True).item()
    
    # 提取数据
    corr_matrices = abide_data['corr']  # 相关矩阵
    pcorr_matrices = abide_data['pcorr']  # 偏相关矩阵
    labels = abide_data['label']
    
    # 如果有 site 信息，也保存
    has_site = 'site' in abide_data
    if has_site:
        sites = abide_data['site']
    
    # 如果有 timeseries 信息，也保存
    has_timeseries = 'timeseries' in abide_data
    if has_timeseries:
        timeseries = abide_data['timeseries']
    
    n_samples = len(labels)
    print(f"总样本数: {n_samples}")
    print(f"节点数: {corr_matrices[0].shape[0]}")
    print(f"标签分布: {dict(Counter(labels))}")
    
    # 验证比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于 1.0"
    
    # 创建索引数组
    indices = np.arange(n_samples)
    
    # 第一次分割：分出训练集和临时集（验证集+测试集）
    temp_ratio = val_ratio + test_ratio
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=labels
    )
    
    # 第二次分割：从临时集中分出验证集和测试集
    val_ratio_adjusted = val_ratio / temp_ratio
    temp_labels = labels[temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_state,
        stratify=temp_labels
    )
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 分割并保存训练集
    print(f"\n处理训练集...")
    train_data = {
        'corr': corr_matrices[train_indices],
        'pcorr': pcorr_matrices[train_indices],
        'label': labels[train_indices]
    }
    if has_site:
        train_data['site'] = sites[train_indices]
    if has_timeseries:
        train_data['timeseries'] = [timeseries[i] for i in train_indices]
    
    train_path = os.path.join(output_dir, 'abide_train.npy')
    np.save(train_path, train_data)
    print(f"训练集保存到: {train_path}")
    print(f"  样本数: {len(train_indices)}")
    print(f"  标签分布: {dict(Counter(labels[train_indices]))}")
    
    # 分割并保存验证集
    print(f"\n处理验证集...")
    val_data = {
        'corr': corr_matrices[val_indices],
        'pcorr': pcorr_matrices[val_indices],
        'label': labels[val_indices]
    }
    if has_site:
        val_data['site'] = sites[val_indices]
    if has_timeseries:
        val_data['timeseries'] = [timeseries[i] for i in val_indices]
    
    val_path = os.path.join(output_dir, 'abide_val.npy')
    np.save(val_path, val_data)
    print(f"验证集保存到: {val_path}")
    print(f"  样本数: {len(val_indices)}")
    print(f"  标签分布: {dict(Counter(labels[val_indices]))}")
    
    # 分割并保存测试集
    print(f"\n处理测试集...")
    test_data = {
        'corr': corr_matrices[test_indices],
        'pcorr': pcorr_matrices[test_indices],
        'label': labels[test_indices]
    }
    if has_site:
        test_data['site'] = sites[test_indices]
    if has_timeseries:
        test_data['timeseries'] = [timeseries[i] for i in test_indices]
    
    test_path = os.path.join(output_dir, 'abide_test.npy')
    np.save(test_path, test_data)
    print(f"测试集保存到: {test_path}")
    print(f"  样本数: {len(test_indices)}")
    print(f"  标签分布: {dict(Counter(labels[test_indices]))}")
    
    print("\n" + "=" * 80)
    print("数据分割完成！")
    print("=" * 80)
    print(f"\n比例检查:")
    print(f"  训练集: {len(train_indices)} ({len(train_indices)/n_samples*100:.1f}%)")
    print(f"  验证集: {len(val_indices)} ({len(val_indices)/n_samples*100:.1f}%)")
    print(f"  测试集: {len(test_indices)} ({len(test_indices)/n_samples*100:.1f}%)")
    print(f"  总计: {len(train_indices) + len(val_indices) + len(test_indices)}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    # 配置路径
    input_file = r'W:\Brain Analysis\FSCN\data\abide.npy'
    output_directory = r'W:\Brain Analysis\FSCN\data'
    
    # 执行分割（7:1:2 比例）
    split_abide_data(
        input_path=input_file,
        output_dir=output_directory,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        random_state=42
    )

