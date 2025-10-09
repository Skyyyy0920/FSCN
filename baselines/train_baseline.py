import os
import sys
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

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.models import get_baseline_model
from data_utils import (
    load_abide_data,
    load_abide_split_data,
    prepare_task_data,
    prepare_task_data_from_dict,
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
    """计算敏感性和特异性"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] < 2:
        return 0.0, 0.0

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return sensitivity, specificity


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for fc, sc, labels in tqdm(dataloader, desc="Training", leave=False):
        fc = fc.to(device)
        sc = sc.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(fc, sc)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        probs_np = probs[:, 1].detach().cpu().numpy()

        valid_mask = labels_np != -1
        all_preds.extend(preds_np[valid_mask])
        all_labels.extend(labels_np[valid_mask])
        all_probs.extend(probs_np[valid_mask])

    avg_loss = total_loss / len(dataloader)

    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        sensitivity, specificity = compute_sensitivity_specificity(all_labels, all_preds)
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except:
            auroc = 0.5
    else:
        accuracy = precision = recall = f1 = sensitivity = specificity = 0.0
        auroc = 0.5

    return avg_loss, accuracy, precision, recall, f1, sensitivity, specificity, auroc


def evaluate(model, dataloader, criterion, device, show_report=False):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for fc, sc, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            fc = fc.to(device)
            sc = sc.to(device)
            labels = labels.to(device)

            logits = model(fc, sc)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()

            valid_mask = labels_np != -1
            all_preds.extend(preds_np[valid_mask])
            all_labels.extend(labels_np[valid_mask])
            all_probs.extend(probs_np[valid_mask])

    avg_loss = total_loss / len(dataloader)

    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        sensitivity, specificity = compute_sensitivity_specificity(all_labels, all_preds)
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except:
            auroc = 0.5

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


def train_baseline_model(model_name, train_data, val_data, test_data, num_nodes, config, device):
    """训练基线模型"""
    print(f"\n{'#' * 80}")
    print(f"# Training Baseline Model: {model_name.upper()}")
    print(f"{'#' * 80}\n")

    print_task_summary(model_name, train_data, val_data, test_data)

    num_classes = get_num_classes(train_data)
    train_labels = [d['label'] for d in train_data]
    train_labels_valid = [l for l in train_labels if l != -1]
    class_counts = dict(Counter(train_labels_valid))

    # 创建数据集和加载器
    train_dataset = BrainDataset(train_data)
    val_dataset = BrainDataset(val_data)
    test_dataset = BrainDataset(test_data) if test_data else None

    if config.get('use_balanced_sampler', False):
        minority_ratio = config.get('minority_ratio', 0.3)
        print(f"Using Balanced Batch Sampler (batch_size={config['batch_size']}, minority_ratio={minority_ratio})")
        train_sampler = BalancedBatchSampler(
            labels=train_labels,
            batch_size=config['batch_size'],
            minority_ratio=minority_ratio,
            ignore_label=-1
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    else:
        print(f"Using standard random sampling (batch_size={config['batch_size']})")
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # 创建模型
    model = get_baseline_model(
        model_name=model_name,
        num_nodes=num_nodes,
        d_model=config['d_model'],
        num_classes=num_classes,
        dropout=config.get('dropout', 0.3)
    ).to(device)

    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数
    print_loss_info(config['loss_type'], gamma=config.get('focal_gamma', 2.0))
    criterion = get_loss_function(
        loss_type=config['loss_type'],
        gamma=config.get('focal_gamma', 2.0),
        class_counts=class_counts,
        ignore_index=-1
    )

    if hasattr(criterion, 'weight') and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)
    if hasattr(criterion, 'alpha') and isinstance(criterion.alpha, torch.Tensor):
        criterion.alpha = criterion.alpha.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 训练循环
    best_val_acc, best_val_sens, best_val_spec = 0, 0, 0
    patience_counter = 0
    best_model_path = f'baselines/best_model_{model_name}.pth'

    print(f"\nTraining {model_name.upper()} model...\n")
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

        # 保存最佳模型
        if val_acc > best_val_acc or (val_sens + val_spec > best_val_sens + best_val_spec):
            best_val_acc = val_acc
            best_val_sens = val_sens
            best_val_spec = val_spec
            patience_counter = 0
            torch.save({
                'model_name': model_name,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, best_model_path)
            print(
                f"  ✓ Saved best model at epoch {epoch + 1} (Acc: {val_acc:.4f}, Sens: {val_sens:.4f}, Spec: {val_spec:.4f})")
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{config['patience']}")

        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
            break

    print(f"Model {model_name.upper()} Training Complete!")

    # 加载最佳模型并评估
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nFinal Evaluation on Validation Set:")
    _, _, _, _, _, _, _, _ = evaluate(model, val_loader, criterion, device, show_report=True)

    # 测试集评估
    test_acc, test_sens, test_spec = 0.0, 0.0, 0.0
    if test_loader is not None:
        print(f"\nFinal Evaluation on Test Set:")
        test_loss, test_acc, test_prec, test_rec, test_f1, test_sens, test_spec, test_auroc = evaluate(
            model, test_loader, criterion, device, show_report=True
        )

        print(f"\n{'*' * 100}")
        print(f"Test Set Performance:")
        print(f"{'*' * 100}")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall: {test_rec:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")
        print(f"  Sensitivity: {test_sens:.4f}")
        print(f"  Specificity: {test_spec:.4f}")
        print(f"  AUROC: {test_auroc:.4f}")
        print(f"{'*' * 100}\n")

    return best_val_acc, best_val_sens, best_val_spec, test_acc, test_sens, test_spec


def load_config_from_yaml(config_path):
    """加载 YAML 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline Model Training')

    # 配置文件
    parser.add_argument('--config', type=str, default='baselines/baseline_config.yml',
                        help='Path to YAML config file')

    # 模型选择
    parser.add_argument('--model', type=str, default='transformer',
                        choices=['gcn', 'transformer', 'gat', 'fc_only', 'sc_only'],
                        help='Baseline model to train')

    # 数据
    parser.add_argument('--dataset', type=str, default='abide', choices=['ABCD', 'abide'])
    parser.add_argument('--use_split_data', action='store_true', default=True)
    parser.add_argument('--train_data_path', type=str, default=r'W:\Brain Analysis\FSCN\data\ABIDE\abide_train.npy')
    parser.add_argument('--val_data_path', type=str, default=r'W:\Brain Analysis\FSCN\data\ABIDE\abide_val.npy')
    parser.add_argument('--test_data_path', type=str, default=r'W:\Brain Analysis\FSCN\data\ABIDE\abide_test.npy')
    parser.add_argument('--data_path', type=str, default=r'W:\Brain Analysis\FSCN\data\ABIDE\abide.npy')
    parser.add_argument('--val_split', type=float, default=0.1)

    # 训练参数
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.3)

    # 其他
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal', 'weighted_ce', 'weighted_focal'])
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])

    args = parser.parse_args()

    # 如果配置文件存在，加载它
    if os.path.exists(args.config):
        yaml_config = load_config_from_yaml(args.config)
        for key, value in yaml_config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

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
    print(f"Baseline Model Training: {args.model.upper()}")
    print("=" * 80)
    for key, value in config.items():
        if key != 'device':
            print(f"  {key}: {value}")
    print(f"  device: {device}")
    print("=" * 80 + "\n")

    # 加载数据
    if args.use_split_data:
        print(f"Loading pre-split ABIDE dataset...")
        train_dict, val_dict, test_dict, num_nodes = load_abide_split_data(
            args.train_data_path,
            args.val_data_path,
            args.test_data_path
        )
        train_data, val_data, test_data = prepare_task_data_from_dict(
            train_dict, val_dict, test_dict, task_idx=None
        )
    else:
        print(f"Loading ABIDE dataset...")
        data_dict, num_nodes = load_abide_data(args.data_path)
        train_data, val_data = prepare_task_data(
            data_dict, task_idx=None, val_split=args.val_split, random_state=args.seed
        )
        test_data = []

    # 训练模型
    best_val_acc, best_val_sens, best_val_spec, test_acc, test_sens, test_spec = train_baseline_model(
        model_name=args.model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        num_nodes=num_nodes,
        config=config,
        device=device
    )

    # 输出最终结果
    print("\n" + "=" * 80)
    print(f"{args.model.upper()} Model Training Complete!")
    print("=" * 80)
    print(f"\nValidation Set (Best Model):")
    print(f"  Accuracy:    {best_val_acc:.4f}")
    print(f"  Sensitivity: {best_val_sens:.4f}")
    print(f"  Specificity: {best_val_spec:.4f}")
    if args.use_split_data:
        print(f"\nTest Set:")
        print(f"  Accuracy:    {test_acc:.4f}")
        print(f"  Sensitivity: {test_sens:.4f}")
        print(f"  Specificity: {test_spec:.4f}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
