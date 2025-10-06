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


def analyze_tasks(data_dict, min_ratio=0.05):
    """
    Analyze sample distribution for all tasks
    
    Note: -1 indicates unlabeled data, which will be ignored when analyzing task validity,
          but the data will not be deleted
    
    Args:
        data_dict: Data dictionary
        min_ratio: Minimum sample ratio threshold (default 5%)
    
    Returns:
        valid_tasks: List of valid tasks
        task_stats: Task statistics information
    """
    print("\n" + "=" * 80)
    print("Dataset Task Analysis")
    print("=" * 80)
    print("Note: Label -1 indicates unlabeled data, which will be ignored in analysis")
    print("      (but data is retained for potential contrastive learning)")

    # Collect labels from all samples
    all_labels = []
    for idx in data_dict:
        labels = data_dict[idx]['label']
        # Ensure labels are in list format
        if not isinstance(labels, list):
            labels = [labels]
        all_labels.append(labels)

    # Detect number of tasks (assuming all samples have same label list length)
    num_tasks = len(all_labels[0])
    print(f"\nTotal number of tasks: {num_tasks}")

    task_stats = {}
    valid_tasks = []

    for task_idx in range(num_tasks):
        # Extract all labels for this task
        task_labels = [labels[task_idx] for labels in all_labels]

        # Count complete class distribution (including -1)
        counter_all = Counter(task_labels)
        total_all = len(task_labels)

        # Count only valid labels (0 and 1) distribution
        task_labels_valid = [label for label in task_labels if label != -1]
        counter_valid = Counter(task_labels_valid)
        total_valid = len(task_labels_valid)

        # Calculate ratios for all classes (for display)
        class_ratios_all = {cls: count / total_all for cls, count in counter_all.items()}

        # Calculate ratios and check balance only based on valid labels (0 and 1)
        if total_valid > 0:
            class_ratios_valid = {cls: count / total_valid for cls, count in counter_valid.items()}
            min_class_ratio_valid = min(class_ratios_valid.values()) if counter_valid else 0
            is_valid = min_class_ratio_valid >= min_ratio and len(counter_valid) >= 2  # At least two classes
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

        # Print statistics
        status = "✓ Valid" if is_valid else "✗ Invalid (imbalanced samples)"
        print(f"\nTask {task_idx}: {status}")
        print(f"  Total samples: {total_all} (labeled: {total_valid}, unlabeled: {counter_all.get(-1, 0)})")
        print(f"  Complete class distribution: {dict(counter_all)}")
        print(f"  Valid class distribution (0,1): {dict(counter_valid)}")
        if total_valid > 0:
            print(
                f"  Valid class ratios: {', '.join([f'Class {k}: {v * 100:.2f}%' for k, v in class_ratios_valid.items()])}")
            print(f"  Minimum valid class ratio: {min_class_ratio_valid * 100:.2f}%")
        else:
            print(f"  Warning: No valid labeled samples!")

    print("\n" + "=" * 80)
    print(f"Number of valid tasks: {len(valid_tasks)}/{num_tasks}")
    print(f"Valid task list: {valid_tasks}")
    print("=" * 80 + "\n")

    return valid_tasks, task_stats


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


def print_task_summary(task_idx, train_data, val_data):
    """
    Print task data summary
    
    Args:
        task_idx: Task index
        train_data: Training data
        val_data: Validation data
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
    print(f"{'=' * 60}\n")
