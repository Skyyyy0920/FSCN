import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
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


class BalancedBatchSampler(Sampler):
    """
    Balanced Batch Sampler with minority class oversampling

    This sampler ensures all majority class samples are used while oversampling
    minority classes to maintain a specified ratio in each batch.

    Args:
        labels: List of labels for all samples
        batch_size: Size of each batch
        minority_ratio: Ratio of minority class samples in each batch (0 to 1)
                       For example, 0.3 means 30% minority, 70% majority
        ignore_label: Label to ignore (e.g., -1 for unlabeled data)
    """

    def __init__(self, labels, batch_size, minority_ratio=0.3, ignore_label=-1):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.minority_ratio = minority_ratio
        self.ignore_label = ignore_label

        # Get indices for each class (excluding ignore_label)
        self.class_indices = {}
        unique_labels = set(labels) - {ignore_label}

        for label in unique_labels:
            self.class_indices[label] = np.where(self.labels == label)[0]

        if len(self.class_indices) == 0:
            raise ValueError("No valid labels found after excluding ignore_label")

        # Identify majority and minority classes
        class_counts = {label: len(indices) for label, indices in self.class_indices.items()}
        self.majority_class = max(class_counts, key=class_counts.get)
        self.minority_classes = [label for label in class_counts.keys() if label != self.majority_class]

        # Calculate samples per batch for each type
        self.minority_samples_per_batch = int(self.batch_size * self.minority_ratio)
        self.majority_samples_per_batch = self.batch_size - self.minority_samples_per_batch

        # If multiple minority classes, distribute minority samples among them
        if len(self.minority_classes) > 0:
            self.samples_per_minority = self.minority_samples_per_batch // len(self.minority_classes)
            self.minority_remainder = self.minority_samples_per_batch % len(self.minority_classes)
        else:
            self.samples_per_minority = 0
            self.minority_remainder = 0

        # Calculate total batches based on majority class (use all majority samples)
        majority_count = len(self.class_indices[self.majority_class])
        self.n_batches = (majority_count + self.majority_samples_per_batch - 1) // self.majority_samples_per_batch

    def __iter__(self):
        # Shuffle majority class indices
        majority_indices = np.random.permutation(self.class_indices[self.majority_class])

        # Oversample minority classes
        minority_indices_oversampled = {}
        for label in self.minority_classes:
            original_indices = self.class_indices[label]
            # Calculate how many samples needed from this minority class
            total_needed = self.samples_per_minority * self.n_batches

            # Oversample by repeating and shuffling
            n_repeats = (total_needed + len(original_indices) - 1) // len(original_indices)
            oversampled = np.tile(original_indices, n_repeats)
            np.random.shuffle(oversampled)
            minority_indices_oversampled[label] = oversampled

        # Generate batches
        for batch_idx in range(self.n_batches):
            batch = []

            # Add majority class samples
            start_maj = batch_idx * self.majority_samples_per_batch
            end_maj = min(start_maj + self.majority_samples_per_batch, len(majority_indices))
            batch.extend(majority_indices[start_maj:end_maj].tolist())

            # Add minority class samples
            for i, label in enumerate(self.minority_classes):
                start_min = batch_idx * self.samples_per_minority
                end_min = start_min + self.samples_per_minority

                # Add remainder to first few minority classes
                if i < self.minority_remainder:
                    end_min += 1

                if end_min <= len(minority_indices_oversampled[label]):
                    batch.extend(minority_indices_oversampled[label][start_min:end_min].tolist())

            # Shuffle the batch to mix different classes
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


def load_raw_data(pickle_path):
    """
    Load raw data from pickle file
    
    Returns:
        data_dict: Raw data dictionary
        num_nodes: Number of nodes/ROIs
    """
    print(f"Loading data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data_dict = pickle.load(f)

    # Get number of nodes
    first_key = list(data_dict.keys())[0]
    num_nodes = data_dict[first_key]['FC'].shape[0]

    print(f"Total samples: {len(data_dict)}")
    print(f"Number of nodes: {num_nodes}")

    return data_dict, num_nodes


def load_abide_data(npy_path):
    """
    Load ABIDE dataset from .npy file
    
    The ABIDE dataset contains:
        - 'timeseries': Time series data
        - 'label': Labels for each sample
        - 'corr': Correlation matrices (used as FC)
        - 'pcorr': Partial correlation matrices (used as SC)
        - 'site': Site information
    
    Args:
        npy_path: Path to the .npy file
    
    Returns:
        data_dict: Converted data dictionary in standard format
        num_nodes: Number of nodes/ROIs
    """
    print(f"Loading ABIDE data from: {npy_path}")

    # Load the .npy file
    abide_data = np.load(npy_path, allow_pickle=True).item()

    # Extract data
    corr_matrices = abide_data['corr']  # Correlation matrices as FC
    pcorr_matrices = abide_data['pcorr']  # Partial correlation matrices as SC
    labels = abide_data['label']

    # TODO
    pcorr_matrices = np.abs(pcorr_matrices)

    # Get number of samples and nodes
    n_samples = len(labels)
    num_nodes = corr_matrices[0].shape[0]

    print(f"Total samples: {n_samples}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Label distribution: {dict(Counter(labels))}")

    # Convert to standard format (similar to original data_dict)
    data_dict = {}
    for idx in range(n_samples):
        # data_dict[idx] = {
        #     'FC': corr_matrices[idx],  # Correlation matrix as FC
        #     'SC': pcorr_matrices[idx],  # Partial correlation matrix as SC
        #     'label': labels[idx]  # Single label (not a list)
        # }
        data_dict[idx] = {
            'FC': pcorr_matrices[idx],  # Correlation matrix as FC
            'SC': pcorr_matrices[idx],  # Partial correlation matrix as SC
            'label': labels[idx]  # Single label (not a list)
        }

    return data_dict, num_nodes


def load_abide_split_data(train_path, val_path, test_path):
    """
    加载已分割的 ABIDE 数据集（训练集、验证集、测试集）
    
    Args:
        train_path: 训练集 .npy 文件路径
        val_path: 验证集 .npy 文件路径
        test_path: 测试集 .npy 文件路径
    
    Returns:
        train_dict: 训练集数据字典
        val_dict: 验证集数据字典
        test_dict: 测试集数据字典
        num_nodes: 节点数
    """
    print(f"Loading split ABIDE data...")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")

    # 加载训练集
    train_data = np.load(train_path, allow_pickle=True).item()
    train_corr = train_data['corr']
    train_pcorr = np.abs(train_data['pcorr'])
    train_labels = train_data['label']

    # 加载验证集
    val_data = np.load(val_path, allow_pickle=True).item()
    val_corr = val_data['corr']
    val_pcorr = np.abs(val_data['pcorr'])
    val_labels = val_data['label']

    # 加载测试集
    test_data = np.load(test_path, allow_pickle=True).item()
    test_corr = test_data['corr']
    test_pcorr = np.abs(test_data['pcorr'])
    test_labels = test_data['label']

    # 获取节点数
    num_nodes = train_corr[0].shape[0]

    print(f"\nDataset statistics:")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Train samples: {len(train_labels)}, distribution: {dict(Counter(train_labels))}")
    print(f"  Val samples: {len(val_labels)}, distribution: {dict(Counter(val_labels))}")
    print(f"  Test samples: {len(test_labels)}, distribution: {dict(Counter(test_labels))}")

    # 转换为标准格式
    train_dict = {}
    for idx in range(len(train_labels)):
        train_dict[idx] = {
            'FC': train_pcorr[idx],
            'SC': train_pcorr[idx],
            'label': train_labels[idx]
        }

    val_dict = {}
    for idx in range(len(val_labels)):
        val_dict[idx] = {
            'FC': val_pcorr[idx],
            'SC': val_pcorr[idx],
            'label': val_labels[idx]
        }

    test_dict = {}
    for idx in range(len(test_labels)):
        test_dict[idx] = {
            'FC': test_pcorr[idx],
            'SC': test_pcorr[idx],
            'label': test_labels[idx]
        }

    return train_dict, val_dict, test_dict, num_nodes


def prepare_task_data(data_dict, task_idx=None, val_split=0.2, random_state=42):
    """
    Prepare training and validation data for a specific task
    
    Args:
        data_dict: Data dictionary
        task_idx: Task index (None for single-task datasets like ABIDE)
        val_split: Validation set split ratio
        random_state: Random seed
    
    Returns:
        train_data: Training data list
        val_data: Validation data list
    """
    # Only use the specific task label
    data_list = []
    for idx in sorted(data_dict.keys()):
        item = data_dict[idx].copy()
        # Extract label for specific task
        all_labels = item['label']

        # Handle multi-task format (list of labels)
        if isinstance(all_labels, list):
            if task_idx is None:
                raise ValueError("task_idx must be specified for multi-task datasets")
            item['label'] = all_labels[task_idx]
        # Handle single-task format (single label) - for ABIDE
        else:
            item['label'] = all_labels

        data_list.append(item)

    labels = [d['label'] for d in data_list]

    train_data, val_data = train_test_split(
        data_list,
        test_size=val_split,
        random_state=random_state,
        stratify=labels
    )

    task_info = f"Task {task_idx}" if task_idx is not None else "Dataset"
    print(f"\n{task_info} Data Split:")
    print(f"  Training set: {len(train_data)} samples")
    train_counter = Counter([d['label'] for d in train_data])
    print(f"    Class distribution: {dict(train_counter)}")

    print(f"  Validation set: {len(val_data)} samples")
    val_counter = Counter([d['label'] for d in val_data])
    print(f"    Class distribution: {dict(val_counter)}")

    return train_data, val_data


def prepare_task_data_from_dict(train_dict, val_dict, test_dict, task_idx=None):
    """
    从预分割的数据字典准备训练、验证和测试数据
    
    Args:
        train_dict: 训练集数据字典
        val_dict: 验证集数据字典
        test_dict: 测试集数据字典
        task_idx: 任务索引（对于单任务数据集如 ABIDE，设为 None）
    
    Returns:
        train_data: 训练数据列表
        val_data: 验证数据列表
        test_data: 测试数据列表
    """

    def convert_to_list(data_dict):
        data_list = []
        for idx in sorted(data_dict.keys()):
            item = data_dict[idx].copy()
            all_labels = item['label']

            # 处理多任务格式（标签列表）
            if isinstance(all_labels, list):
                if task_idx is None:
                    raise ValueError("task_idx must be specified for multi-task datasets")
                item['label'] = all_labels[task_idx]
            # 处理单任务格式（单个标签）- 用于 ABIDE
            else:
                item['label'] = all_labels

            data_list.append(item)
        return data_list

    train_data = convert_to_list(train_dict)
    val_data = convert_to_list(val_dict)
    test_data = convert_to_list(test_dict)

    task_info = f"Task {task_idx}" if task_idx is not None else "Dataset"
    print(f"\n{task_info} Data from pre-split files:")
    print(f"  Training set: {len(train_data)} samples")
    train_counter = Counter([d['label'] for d in train_data])
    print(f"    Class distribution: {dict(train_counter)}")

    print(f"  Validation set: {len(val_data)} samples")
    val_counter = Counter([d['label'] for d in val_data])
    print(f"    Class distribution: {dict(val_counter)}")

    print(f"  Test set: {len(test_data)} samples")
    test_counter = Counter([d['label'] for d in test_data])
    print(f"    Class distribution: {dict(test_counter)}")

    return train_data, val_data, test_data


def get_num_classes(data_list):
    """
    Get number of classes in dataset (excluding -1)
    
    Args:
        data_list: List of data samples
    
    Returns:
        Number of unique classes (excluding -1 for unlabeled)
    """
    labels = [d['label'] for d in data_list]
    unique_labels = set(labels)
    # Remove -1 (unlabeled class)
    unique_labels.discard(-1)
    return len(unique_labels)


def print_task_summary(task_idx, train_data, val_data, test_data=None):
    """
    Print task data summary
    
    Args:
        task_idx: Task index
        train_data: Training data
        val_data: Validation data
        test_data: Test data (optional)
    """
    train_labels = [d['label'] for d in train_data]
    val_labels = [d['label'] for d in val_data]

    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)

    # Count valid labels (0 and 1)
    train_labels_valid = [l for l in train_labels if l != -1]
    val_labels_valid = [l for l in val_labels if l != -1]
    train_counter_valid = Counter(train_labels_valid)
    val_counter_valid = Counter(val_labels_valid)

    print(f"\n{'=' * 60}")
    print(f"Task {task_idx} Data Summary")
    print(f"{'=' * 60}")
    print(f"Training set: {len(train_data)} samples")
    print(f"  Complete distribution: {dict(train_counter)}")
    print(f"  Valid labels (0,1): {dict(train_counter_valid)} ({len(train_labels_valid)} samples)")
    print(f"Validation set: {len(val_data)} samples")
    print(f"  Complete distribution: {dict(val_counter)}")
    print(f"  Valid labels (0,1): {dict(val_counter_valid)} ({len(val_labels_valid)} samples)")

    if test_data is not None:
        test_labels = [d['label'] for d in test_data]
        test_counter = Counter(test_labels)
        test_labels_valid = [l for l in test_labels if l != -1]
        test_counter_valid = Counter(test_labels_valid)
        print(f"Test set: {len(test_data)} samples")
        print(f"  Complete distribution: {dict(test_counter)}")
        print(f"  Valid labels (0,1): {dict(test_counter_valid)} ({len(test_labels_valid)} samples)")

    print(f"{'=' * 60}\n")
