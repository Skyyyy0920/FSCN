"""
对比所有基线模型的结果
从训练日志中提取性能指标并生成对比表格
"""
import os
import re
import argparse
from datetime import datetime
import glob


def extract_metrics_from_log(log_file):
    """从日志文件中提取性能指标"""
    metrics = {
        'val_acc': None, 'val_sens': None, 'val_spec': None,
        'test_acc': None, 'test_sens': None, 'test_spec': None,
        'test_f1': None, 'test_auroc': None
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 提取验证集最佳指标
            val_match = re.search(
                r'Validation Set \(Best Model\):\s+Accuracy:\s+([\d.]+)\s+Sensitivity:\s+([\d.]+)\s+Specificity:\s+([\d.]+)',
                content, re.MULTILINE
            )
            if val_match:
                metrics['val_acc'] = float(val_match.group(1))
                metrics['val_sens'] = float(val_match.group(2))
                metrics['val_spec'] = float(val_match.group(3))
            
            # 提取测试集指标
            test_match = re.search(
                r'Test Set:\s+Accuracy:\s+([\d.]+)\s+Sensitivity:\s+([\d.]+)\s+Specificity:\s+([\d.]+)',
                content, re.MULTILINE
            )
            if test_match:
                metrics['test_acc'] = float(test_match.group(1))
                metrics['test_sens'] = float(test_match.group(2))
                metrics['test_spec'] = float(test_match.group(3))
            
            # 提取测试集 F1 和 AUROC
            perf_match = re.search(
                r'F1-Score:\s+([\d.]+).*?AUROC:\s+([\d.]+)',
                content, re.DOTALL
            )
            if perf_match:
                metrics['test_f1'] = float(perf_match.group(1))
                metrics['test_auroc'] = float(perf_match.group(2))
    
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return metrics


def find_latest_logs(log_dir, models):
    """找到每个模型的最新日志文件"""
    latest_logs = {}
    
    for model in models:
        pattern = os.path.join(log_dir, f'{model}_*.log')
        log_files = glob.glob(pattern)
        
        if log_files:
            # 按修改时间排序，取最新的
            latest_log = max(log_files, key=os.path.getmtime)
            latest_logs[model] = latest_log
        else:
            print(f"Warning: No log file found for {model}")
    
    return latest_logs


def print_comparison_table(results):
    """打印对比表格"""
    print("\n" + "=" * 120)
    print("BASELINE MODELS PERFORMANCE COMPARISON")
    print("=" * 120)
    
    # 验证集性能
    print("\n验证集性能 (Validation Set - Best Model):")
    print("-" * 120)
    print(f"{'Model':<20} {'Accuracy':<15} {'Sensitivity':<15} {'Specificity':<15} {'Avg':<15}")
    print("-" * 120)
    
    for model, metrics in sorted(results.items()):
        if metrics['val_acc'] is not None:
            avg = (metrics['val_acc'] + metrics['val_sens'] + metrics['val_spec']) / 3
            print(f"{model.upper():<20} {metrics['val_acc']:<15.4f} {metrics['val_sens']:<15.4f} {metrics['val_spec']:<15.4f} {avg:<15.4f}")
        else:
            print(f"{model.upper():<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("-" * 120)
    
    # 测试集性能
    print("\n测试集性能 (Test Set):")
    print("-" * 120)
    print(f"{'Model':<20} {'Accuracy':<15} {'Sensitivity':<15} {'Specificity':<15} {'F1-Score':<15} {'AUROC':<15}")
    print("-" * 120)
    
    for model, metrics in sorted(results.items()):
        if metrics['test_acc'] is not None:
            print(f"{model.upper():<20} {metrics['test_acc']:<15.4f} {metrics['test_sens']:<15.4f} {metrics['test_spec']:<15.4f} {metrics.get('test_f1', 0):<15.4f} {metrics.get('test_auroc', 0):<15.4f}")
        else:
            print(f"{model.upper():<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("-" * 120)
    
    # 计算最佳模型
    print("\n最佳模型排名 (按测试集准确率):")
    print("-" * 120)
    
    valid_results = [(model, m) for model, m in results.items() if m['test_acc'] is not None]
    if valid_results:
        sorted_results = sorted(valid_results, key=lambda x: x[1]['test_acc'], reverse=True)
        
        for rank, (model, metrics) in enumerate(sorted_results, 1):
            print(f"{rank}. {model.upper():<18} Acc: {metrics['test_acc']:.4f}  Sens: {metrics['test_sens']:.4f}  Spec: {metrics['test_spec']:.4f}")
    else:
        print("No valid results found.")
    
    print("=" * 120 + "\n")


def save_results_to_csv(results, output_file):
    """保存结果到 CSV 文件"""
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow([
            'Model', 
            'Val_Acc', 'Val_Sens', 'Val_Spec',
            'Test_Acc', 'Test_Sens', 'Test_Spec', 'Test_F1', 'Test_AUROC'
        ])
        
        # 写入数据
        for model, metrics in sorted(results.items()):
            writer.writerow([
                model,
                metrics.get('val_acc', ''),
                metrics.get('val_sens', ''),
                metrics.get('val_spec', ''),
                metrics.get('test_acc', ''),
                metrics.get('test_sens', ''),
                metrics.get('test_spec', ''),
                metrics.get('test_f1', ''),
                metrics.get('test_auroc', '')
            ])
    
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare baseline model results')
    parser.add_argument('--log_dir', type=str, default='baselines/logs',
                        help='Directory containing log files')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['gcn', 'transformer', 'gat', 'fc_only', 'sc_only'],
                        help='Models to compare')
    parser.add_argument('--output', type=str, default='baselines/comparison_results.csv',
                        help='Output CSV file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXTRACTING BASELINE MODEL RESULTS")
    print("=" * 80)
    print(f"\nLog directory: {args.log_dir}")
    print(f"Models: {args.models}\n")
    
    # 找到最新的日志文件
    latest_logs = find_latest_logs(args.log_dir, args.models)
    
    if not latest_logs:
        print("Error: No log files found!")
        return
    
    print("Latest log files:")
    for model, log_file in latest_logs.items():
        print(f"  {model}: {log_file}")
    print()
    
    # 提取指标
    results = {}
    for model, log_file in latest_logs.items():
        print(f"Extracting metrics from {model}...")
        metrics = extract_metrics_from_log(log_file)
        results[model] = metrics
    
    # 打印对比表格
    print_comparison_table(results)
    
    # 保存到 CSV
    save_results_to_csv(results, args.output)


if __name__ == '__main__':
    main()

