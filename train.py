import os
import argparse
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tqdm import tqdm
from collections import Counter

from model import DualBranchModel
from data_utils import (
    load_raw_data,
    load_abide_data,
    analyze_tasks,
    prepare_task_data,
    get_num_classes,
    print_task_summary,
    BrainDataset,
    BalancedBatchSampler
)
from losses import get_loss_function, print_loss_info


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_sensitivity_specificity(y_true, y_pred):
    """
    Compute Sensitivity and Specificity from predictions

    Sensitivity (Recall/TPR) = TP / (TP + FN) - ability to detect positive cases
    Specificity (TNR) = TN / (TN + FP) - ability to detect negative cases

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        sensitivity, specificity
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] < 2:
        return 0.0, 0.0

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return sensitivity, specificity


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for fc, sc, labels in tqdm(dataloader, desc="Training FSCN", leave=False):
        fc = fc.to(device)
        sc = sc.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(fc, sc)
        loss = criterion(logits, labels)  # criterion should set ignore_index=-1

        # Backward pass
        loss.backward()
        optimizer.step()

        # Record metrics (only for samples with valid labels)
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Convert to numpy
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        probs_np = probs[:, 1].detach().cpu().numpy()

        # Only keep samples with valid labels (!= -1) for metric calculation
        valid_mask = labels_np != -1
        all_preds.extend(preds_np[valid_mask])
        all_labels.extend(labels_np[valid_mask])
        all_probs.extend(probs_np[valid_mask])

    avg_loss = total_loss / len(dataloader)

    # Compute metrics (only based on valid labels)
    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        sensitivity, specificity = compute_sensitivity_specificity(all_labels, all_preds)
        # Compute AUROC (requires at least two classes)
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except:
            auroc = 0.5
    else:
        accuracy = precision = recall = f1 = sensitivity = specificity = 0.0
        auroc = 0.5

    return avg_loss, accuracy, precision, recall, f1, sensitivity, specificity, auroc


def evaluate(model, dataloader, criterion, device, show_report=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for fc, sc, labels in tqdm(dataloader, desc="Validating", leave=False):
            fc = fc.to(device)
            sc = sc.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(fc, sc)
            loss = criterion(logits, labels)  # criterion should set ignore_index=-1

            # Record metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Convert to numpy
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()

            # Only keep samples with valid labels (!= -1) for metric calculation
            valid_mask = labels_np != -1
            all_preds.extend(preds_np[valid_mask])
            all_labels.extend(labels_np[valid_mask])
            all_probs.extend(probs_np[valid_mask])

    avg_loss = total_loss / len(dataloader)

    # Compute metrics (only based on valid labels)
    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        sensitivity, specificity = compute_sensitivity_specificity(all_labels, all_preds)
        # Compute AUROC
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except:
            auroc = 0.5

        # Show detailed report if requested
        if show_report:
            print("\n" + "=" * 60)
            print("Classification Report:")
            print("=" * 60)
            print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'], zero_division=0))
            cm = confusion_matrix(all_labels, all_preds)
            print("\nConfusion Matrix:")
            print(f"                 Predicted")
            print(f"              Neg      Pos")
            print(f"Actual Neg   {cm[0, 0]:4d}    {cm[0, 1]:4d}")
            print(f"       Pos   {cm[1, 0]:4d}    {cm[1, 1]:4d}")
            print(f"\nSensitivity (Recall/TPR): {sensitivity:.4f}")
            print(f"Specificity (TNR):        {specificity:.4f}")
            print("=" * 60 + "\n")
    else:
        accuracy = precision = recall = f1 = sensitivity = specificity = 0.0
        auroc = 0.5

    return avg_loss, accuracy, precision, recall, f1, sensitivity, specificity, auroc


def train_single_task(task_idx, train_data, val_data, num_nodes, config, device):
    print(f"\n{'#' * 80}")
    print(f"# Starting Training for Task {task_idx}")
    print(f"{'#' * 80}\n")

    print_task_summary(task_idx, train_data, val_data)

    num_classes = get_num_classes(train_data)

    train_labels = [d['label'] for d in train_data]
    train_labels_valid = [l for l in train_labels if l != -1]  # filter label = -1
    class_counts = dict(Counter(train_labels_valid))

    train_dataset = BrainDataset(train_data)
    val_dataset = BrainDataset(val_data)

    # Use Balanced Batch Sampler for training to handle class imbalance
    if config.get('use_balanced_sampler', True):
        minority_ratio = config.get('minority_ratio', 0.3)
        print(f"Using Balanced Batch Sampler (batch_size={config['batch_size']}, minority_ratio={minority_ratio})")
        train_sampler = BalancedBatchSampler(
            labels=train_labels,
            batch_size=config['batch_size'],
            minority_ratio=minority_ratio,
            ignore_label=-1
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0
        )
    else:
        print(f"Using standard random sampling (batch_size={config['batch_size']})")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    model = DualBranchModel(
        num_nodes=num_nodes,
        d_model=config['d_model'],
        num_classes=num_classes
    ).to(device)

    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")

    print_loss_info(config['loss_type'], gamma=config.get('focal_gamma', 2.0))

    criterion = get_loss_function(
        loss_type=config['loss_type'],
        gamma=config.get('focal_gamma', 2.0),
        class_counts=class_counts,
        ignore_index=-1  # Ignore unlabeled samples
    )

    if hasattr(criterion, 'weight') and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)
    if hasattr(criterion, 'alpha') and isinstance(criterion.alpha, torch.Tensor):
        criterion.alpha = criterion.alpha.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    best_val_f1 = 0
    best_val_auroc = 0
    patience_counter = 0
    best_model_path = f'best_model_task_{task_idx}.pth'

    print(f"\nTraining Task {task_idx}...\n")
    for epoch in range(config['epochs']):
        train_loss, train_acc, train_prec, train_rec, train_f1, train_sens, train_spec, train_auroc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, val_sens, val_spec, val_auroc = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Epoch [{epoch + 1}/{config['epochs']}]")
        print(
            f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, Sens: {train_sens:.4f}, Spec: {train_spec:.4f}, AUROC: {train_auroc:.4f}")
        print(
            f"  Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, Sens: {val_sens:.4f}, Spec: {val_spec:.4f}, AUROC: {val_auroc:.4f}")

        # Use F1-score as primary metric for model selection (better for imbalanced data)
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_auroc > best_val_auroc):
            best_val_f1 = val_f1
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save({
                'task_idx': task_idx,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_auroc': val_auroc,
                'config': config,
                'num_classes': num_classes
            }, best_model_path)
            print(f"  ✓ Saved best model (F1: {val_f1:.4f}, AUROC: {val_auroc:.4f})")
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{config['patience']}")

        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
            break

        print()

    # Load best model and show final evaluation report
    print(f"\n{'=' * 80}")
    print(f"Task {task_idx} Training Complete!")
    print(f"{'=' * 80}")
    print(f"Best validation F1-score: {best_val_f1:.4f}")
    print(f"Best validation AUROC: {best_val_auroc:.4f}")
    print(f"Best model saved to: {best_model_path}")

    # Load best model for final evaluation
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Show detailed classification report on validation set
    print(f"\nFinal Evaluation on Validation Set:")
    _, _, _, _, _, _, _, _ = evaluate(model, val_loader, criterion, device, show_report=True)

    return best_val_f1, best_val_auroc


def load_config_from_yaml(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description='FSCN experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration file
    parser.add_argument('--config', type=str, default='args/abide.yml', help='Path to YAML config file.')

    # Data
    parser.add_argument('--dataset', type=str, default='abide', choices=['ABCD', 'abide'],
                        help='Dataset to use: ABCD (multi-task) or abide (single-task)')
    parser.add_argument('--data_path', type=str,
                        # default=r'W:\Brain Analysis\data\data\data_dict.pkl',
                        default=r'W:\Brain Analysis\FSCN\data\abide.npy',
                        help='Path to data file (.pkl for ABCD, .npy for ABIDE)')
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--min_ratio', type=float, default=0.05,
                        help='Minimum sample ratio threshold (for filtering imbalanced tasks)')

    # Model
    parser.add_argument('--d_model', type=int, default=128, help='Model hidden size')

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--use_balanced_sampler', action='store_true', default=False,
                        help='Use balanced batch sampler to handle class imbalance')
    parser.add_argument('--minority_ratio', type=float, default=0.3,
                        help='Ratio of minority class samples in each batch (0 to 1)')

    # Loss Function
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal', 'weighted_ce', 'weighted_focal'],
                        help='Loss function type')
    parser.add_argument('--focal_gamma', type=float, default=3.0, help='Focal Loss gamma parameter')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--tasks', type=int, nargs='+', default=None,
                        help='Specify task IDs to train (only for multi-task datasets, default: [4,5,7,8] for ABCD)')

    args = parser.parse_args()
    yaml_config = load_config_from_yaml(args.config)
    for key, value in yaml_config.items():
        setattr(args, key, value)

    # Set default tasks for ABCD dataset
    if args.tasks is None and args.dataset == 'ABCD':
        args.tasks = [4, 5, 7, 8]

    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    config = vars(args)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("Training config:")
    print("=" * 80)
    for key, value in config.items():
        if key != 'device':
            print(f"  {key}: {value}")
    print(f"  device: {device}")
    print("=" * 80 + "\n")

    # Load data based on dataset type
    if args.dataset == 'abide':
        print(f"Loading ABIDE dataset...")
        data_dict, num_nodes = load_abide_data(args.data_path)
    else:
        print(f"Loading ABCD dataset...")
        data_dict, num_nodes = load_raw_data(args.data_path)

    ###################################################################################
    # valid_tasks, task_stats = analyze_tasks(data_dict, min_ratio=args.min_ratio)

    # if len(valid_tasks) == 0:
    #     print("Error: No valid tasks found!")
    #     return

    # if args.tasks is not None:
    #     specified_tasks = [t for t in args.tasks if t in valid_tasks]
    #     if len(specified_tasks) == 0:
    #         print(f"Error: None of the specified tasks {args.tasks} are valid!")
    #         return
    #     valid_tasks = specified_tasks
    #     print(f"\nOnly training tasks: {valid_tasks}\n")
    ###################################################################################

    results = {}

    # Handle single-task dataset (ABIDE)
    if args.dataset == 'abide':
        print("\nTraining ABIDE dataset (single-task binary classification)...\n")

        train_data, val_data = prepare_task_data(
            data_dict,
            task_idx=None,  # No task index for single-task dataset
            val_split=args.val_split,
            random_state=args.seed
        )

        best_f1, best_auroc = train_single_task(
            task_idx='abide',  # Use 'abide' as identifier
            train_data=train_data,
            val_data=val_data,
            num_nodes=num_nodes,
            config=config,
            device=device
        )

        results['abide'] = {'f1': best_f1, 'auroc': best_auroc}

        print("\n" + "=" * 80)
        print("ABIDE Dataset Training Complete!")
        print("=" * 80)
        print(f"\nBest F1-score:  {best_f1:.4f}")
        print(f"Best AUROC:     {best_auroc:.4f}")
        print("=" * 80 + "\n")

    # Handle multi-task dataset (ABCD)
    else:
        for task_idx in args.tasks:
            train_data, val_data = prepare_task_data(
                data_dict,
                task_idx=task_idx,
                val_split=args.val_split,
                random_state=args.seed
            )

            best_f1, best_auroc = train_single_task(
                task_idx=task_idx,
                train_data=train_data,
                val_data=val_data,
                num_nodes=num_nodes,
                config=config,
                device=device
            )

            results[task_idx] = {'f1': best_f1, 'auroc': best_auroc}

        print("\n" + "=" * 80)
        print("All Tasks Training Complete!")
        print("=" * 80)
        print("\nTask Performance Summary:")
        print(f"{'Task ID':<10} {'Best F1':<15} {'Best AUROC':<15} {'Status'}")
        print("-" * 60)
        for task_idx in args.tasks:
            f1 = results[task_idx]['f1']
            auroc = results[task_idx]['auroc']
            print(f"{task_idx:<10} {f1:<15.4f} {auroc:<15.4f} FINISHED")

        avg_f1 = np.mean([r['f1'] for r in results.values()])
        avg_auroc = np.mean([r['auroc'] for r in results.values()])
        print("-" * 60)
        print(f"{'Average':<10} {avg_f1:<15.4f} {avg_auroc:<15.4f}")
        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
