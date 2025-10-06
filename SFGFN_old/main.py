import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import (
    load_data_dict,
    create_data_splits,
    BrainNetworkDataset,
    collate_fn,
    get_num_classes
)
from models import BrainNetworkFusionModel
from trainer import Trainer


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='脑网络多模态融合实验')

    # data param
    parser.add_argument('--data_path', type=str, default=r'W:\Brain Analysis\data\data\data_dict.pkl')
    parser.add_argument('--threshold', type=float, default=0.1)

    # model param
    parser.add_argument('--fusion_method', type=str, default='all', choices=['A', 'B', 'C', 'all'])
    parser.add_argument('--node_feature_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--pooling_type', type=str, default='mean', choices=['mean', 'sum', 'attention'])
    parser.add_argument('--task_type', type=str, default='classification', choices=['classification', 'regression'])

    # training param
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--early_stopping_patience', type=int, default=20)

    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results')

    return parser.parse_args()


def run_experiment(args, fusion_method):
    print(f"\n{'=' * 80}")
    print(f"运行融合方法 {fusion_method}")
    print(f"{'=' * 80}\n")

    method_save_dir = os.path.join(args.save_dir, f'method_{fusion_method}')
    os.makedirs(method_save_dir, exist_ok=True)

    data_dict = load_data_dict(args.data_path)
    num_classes = get_num_classes(data_dict)

    first_key = list(data_dict.keys())[0]
    num_nodes = data_dict[first_key]['FC'].shape[0]
    print(f"ROI num: {num_nodes}")

    train_indices, val_indices, test_indices = create_data_splits(data_dict)

    train_dataset = BrainNetworkDataset(data_dict, train_indices, args.threshold)
    val_dataset = BrainNetworkDataset(data_dict, val_indices, args.threshold)
    test_dataset = BrainNetworkDataset(data_dict, test_indices, args.threshold)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    model = BrainNetworkFusionModel(
        num_nodes=num_nodes,
        node_feature_dim=args.node_feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        fusion_method=fusion_method,
        num_gnn_layers=args.num_gnn_layers,
        pooling_type=args.pooling_type,
        task_type=args.task_type
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model param num: {num_params:,}")

    device = args.device if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        task_type=args.task_type,
        num_classes=num_classes,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=method_save_dir
    )

    results = trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience
    )

    results['fusion_method'] = fusion_method
    results['hyperparameters'] = vars(args)

    return results


def compare_results(all_results, save_dir):
    print(f"\n{'=' * 80}")
    print("结果比较")
    print(f"{'=' * 80}\n")

    comparison = {}

    for result in all_results:
        method = result['fusion_method']
        comparison[method] = {
            'best_epoch': result['best_epoch'],
            'best_val_loss': result['best_val_loss'],
            'test_loss': result['test_loss']
        }

        # 添加任务特定指标
        for key, value in result.items():
            if key.startswith('test_') and key != 'test_loss':
                comparison[method][key] = value

    # 打印比较表格
    print(f"{'方法':<10} {'最佳Epoch':<12} {'验证损失':<12} {'测试损失':<12}", end="")

    # 获取所有指标名称
    metric_names = set()
    for method_results in comparison.values():
        metric_names.update([k for k in method_results.keys()
                             if k.startswith('test_') and k != 'test_loss'])
    metric_names = sorted(metric_names)

    for metric in metric_names:
        print(f"{metric.replace('test_', '').upper():<12}", end="")
    print()
    print("-" * (46 + 12 * len(metric_names)))

    for method, metrics in comparison.items():
        print(f"Method {method:<3} {metrics['best_epoch']:<12} "
              f"{metrics['best_val_loss']:<12.4f} {metrics['test_loss']:<12.4f}", end="")
        for metric in metric_names:
            if metric in metrics:
                print(f"{metrics[metric]:<12.4f}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()

    # 保存比较结果
    comparison_file = os.path.join(save_dir, 'comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=4)
    print(f"\n比较结果已保存到: {comparison_file}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    config_file = os.path.join(args.save_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=4)

    if args.fusion_method == 'all':
        methods = ['A', 'B', 'C']
        all_results = []

        for method in methods:
            result = run_experiment(args, method)
            all_results.append(result)

        compare_results(all_results, args.save_dir)
    else:
        run_experiment(args, args.fusion_method)

    print(f"\n{'=' * 80}")
    print("实验完成!")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
