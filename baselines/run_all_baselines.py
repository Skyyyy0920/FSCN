"""
运行所有基线模型的脚本
用于批量对比不同基线模型的性能
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_baseline(model_name, config_path, log_dir='baselines/logs'):
    """运行单个基线模型"""
    print(f"\n{'=' * 80}")
    print(f"Running baseline: {model_name.upper()}")
    print(f"{'=' * 80}\n")
    
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{model_name}_{timestamp}.log')
    
    # 运行训练命令
    cmd = [
        'python', 'baselines/train_baseline.py',
        '--config', config_path,
        '--model', model_name
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}\n")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8'
        )
        
        for line in process.stdout:
            print(line, end='')
            f.write(line)
        
        process.wait()
    
    return process.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run all baseline models')
    parser.add_argument('--config', type=str, default='baselines/baseline_config.yml',
                        help='Path to config file')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['gcn', 'transformer', 'gat', 'fc_only', 'sc_only'],
                        help='Baseline models to run')
    parser.add_argument('--log_dir', type=str, default='baselines/logs',
                        help='Directory to save logs')
    args = parser.parse_args()
    
    print("=" * 80)
    print("RUNNING ALL BASELINE MODELS")
    print("=" * 80)
    print(f"\nModels to run: {args.models}")
    print(f"Config file: {args.config}")
    print(f"Log directory: {args.log_dir}\n")
    
    results = {}
    
    for model_name in args.models:
        success = run_baseline(model_name, args.config, args.log_dir)
        results[model_name] = 'SUCCESS' if success else 'FAILED'
        
        print(f"\n{model_name.upper()}: {results[model_name]}")
    
    # 打印汇总结果
    print("\n" + "=" * 80)
    print("BASELINE MODELS TRAINING SUMMARY")
    print("=" * 80)
    for model_name, status in results.items():
        status_symbol = '✓' if status == 'SUCCESS' else '✗'
        print(f"  {status_symbol} {model_name.upper():<15} {status}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()

