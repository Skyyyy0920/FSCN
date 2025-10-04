import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from collections import Counter

from model import DualBranchModel
from data_utils import (
    load_raw_data,
    analyze_tasks,
    prepare_task_data,
    get_num_classes,
    print_task_summary,
    BrainDataset
)
from losses import get_loss_function, print_loss_info

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

        # 前向传播
        optimizer.zero_grad()
        logits = model(fc, sc)
        loss = criterion(logits, labels)  # criterion应该设置ignore_index=-1

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录（只记录有效标签的样本用于指标计算）
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # 转换为numpy
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        probs_np = probs[:, 1].detach().cpu().numpy()

        # 只保留有效标签（!= -1）的样本用于指标计算
        valid_mask = labels_np != -1
        all_preds.extend(preds_np[valid_mask])
        all_labels.extend(labels_np[valid_mask])
        all_probs.extend(probs_np[valid_mask])

    avg_loss = total_loss / len(dataloader)

    # 计算准确率和AUROC（只基于有效标签）
    if len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        # 计算AUROC（需要至少两个类别）
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except:
            auroc = 0.5
    else:
        accuracy = 0.0
        auroc = 0.5

    return avg_loss, accuracy, auroc


def evaluate(model, dataloader, criterion, device):
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

            # 前向传播
            logits = model(fc, sc)
            loss = criterion(logits, labels)  # criterion应该设置ignore_index=-1

            # 记录
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # 转换为numpy
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()

            # 只保留有效标签（!= -1）的样本用于指标计算
            valid_mask = labels_np != -1
            all_preds.extend(preds_np[valid_mask])
            all_labels.extend(labels_np[valid_mask])
            all_probs.extend(probs_np[valid_mask])

    avg_loss = total_loss / len(dataloader)

    # 计算准确率和AUROC（只基于有效标签）
    if len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        # 计算AUROC
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except:
            auroc = 0.5
    else:
        accuracy = 0.0
        auroc = 0.5

    return avg_loss, accuracy, auroc


def train_single_task(task_idx, train_data, val_data, num_nodes, config, device):
    print(f"\n{'#' * 80}")
    print(f"# 开始训练任务 {task_idx}")
    print(f"{'#' * 80}\n")

    print_task_summary(task_idx, train_data, val_data)

    num_classes = get_num_classes(train_data)
    print(f"num of class: {num_classes}")

    train_labels = [d['label'] for d in train_data]
    train_labels_valid = [l for l in train_labels if l != -1]  # filter label = -1
    class_counts = dict(Counter(train_labels_valid))

    print(f"Training data distribution: {dict(Counter(train_labels))}")
    print(f"label distribution (0,1): {class_counts}")

    train_dataset = BrainDataset(train_data)
    val_dataset = BrainDataset(val_data)

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

    print(f"Model parameter num: {sum(p.numel() for p in model.parameters()):,}")

    print_loss_info(config['loss_type'], gamma=config.get('focal_gamma', 2.0))

    criterion = get_loss_function(
        loss_type=config['loss_type'],
        gamma=config.get('focal_gamma', 2.0),
        class_counts=class_counts,
        ignore_index=-1  # 忽略无标签样本
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

    best_val_auroc = 0
    patience_counter = 0
    best_model_path = f'best_model_task_{task_idx}.pth'

    print(f"\nTraining task {task_idx}...\n")
    for epoch in range(config['epochs']):
        train_loss, train_acc, train_auroc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_auroc = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Epoch [{epoch + 1}/{config['epochs']}]")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUROC: {train_auroc:.4f}")
        print(f"  Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUROC: {val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save({
                'task_idx': task_idx,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'config': config,
                'num_classes': num_classes
            }, best_model_path)
            print(f"  ✓ 保存最佳模型 (AUROC: {val_auroc:.4f})")
        else:
            patience_counter += 1
            print(f"  Early Stopping count: {patience_counter}/{config['patience']}")

        if patience_counter >= config['patience']:
            print(f"\nEarly Stopping at epoch {epoch + 1}!")
            break

        print()

    print(f"\n任务 {task_idx} 训练完成！")
    print(f"最佳验证AUROC: {best_val_auroc:.4f}")
    print(f"最佳模型已保存至: {best_model_path}\n")

    return best_val_auroc


def parse_args():
    parser = argparse.ArgumentParser(
        description='FSCN experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # data
    parser.add_argument('--data_path', type=str,
                        default=r'W:\Brain Analysis\data\data\data_dict.pkl')
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--min_ratio', type=float, default=0.05,
                        help='最小样本比例阈值（用于过滤不平衡任务）')

    # Model
    parser.add_argument('--d_model', type=int, default=128, help='model hidden size')

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    # Loss Function
    parser.add_argument('--loss_type', type=str, default='weighted_ce',
                        choices=['ce', 'focal', 'weighted_ce', 'weighted_focal'],
                        help='loss function type')
    parser.add_argument('--focal_gamma', type=float, default=3.0, help='Focal Loss gamma')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--tasks', type=int, nargs='+', default=[4, 5, 7, 8],
                        help='指定要训练的任务ID（默认训练所有有效任务）')

    args = parser.parse_args()
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

    data_dict, num_nodes = load_raw_data(args.data_path)

    ###################################################################################
    # valid_tasks, task_stats = analyze_tasks(data_dict, min_ratio=args.min_ratio)

    # if len(valid_tasks) == 0:
    #     print("错误：没有找到有效的任务！")
    #     return

    # if args.tasks is not None:
    #     specified_tasks = [t for t in args.tasks if t in valid_tasks]
    #     if len(specified_tasks) == 0:
    #         print(f"Error: None of the specified tasks {args.tasks} are valid!")
    #         return
    #     valid_tasks = specified_tasks
    #     print(f"\nOnly train task: {valid_tasks}\n")
    ###################################################################################

    results = {}

    for task_idx in args.tasks:
        train_data, val_data = prepare_task_data(
            data_dict,
            task_idx,
            val_split=args.val_split,
            random_state=args.seed
        )

        best_auroc = train_single_task(
            task_idx,
            train_data,
            val_data,
            num_nodes,
            config,
            device
        )

        results[task_idx] = best_auroc

    print("\n" + "=" * 80)
    print("所有任务训练完成！")
    print("=" * 80)
    print("\n任务性能总结:")
    print(f"{'任务ID':<10} {'最佳AUROC':<15} {'状态'}")
    print("-" * 40)
    for task_idx in args.tasks:
        auroc = results[task_idx]
        print(f"{task_idx:<10} {auroc:<15.4f} FINISHED")

    avg_auroc = np.mean(list(results.values()))
    print("-" * 40)
    print(f"{'平均':<10} {avg_auroc:<15.4f}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
