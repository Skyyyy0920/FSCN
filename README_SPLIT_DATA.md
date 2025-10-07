# ABIDE 数据集分割和测试集使用说明

## 概述

本项目现在支持将 ABIDE 数据集按照 7:1:2 的比例分割成训练集、验证集和测试集，并在训练过程中对测试集进行评估。

## 步骤 1: 分割数据

首先运行 `split_data.py` 脚本将原始的 `abide.npy` 文件分割成三个独立的文件：

```bash
python split_data.py
```

该脚本会生成以下文件：
- `data/abide_train.npy` - 训练集（70%）
- `data/abide_val.npy` - 验证集（10%）
- `data/abide_test.npy` - 测试集（20%）

分割过程会保持类别平衡（stratified split），确保每个子集中都有相似的类别分布。

## 步骤 2: 配置训练

在 `args/abide.yml` 配置文件中，设置以下参数：

```yaml
# 使用预分割的数据文件
use_split_data: true

# 指定分割后的文件路径
train_data_path: W:\Brain Analysis\FSCN\data\abide_train.npy
val_data_path: W:\Brain Analysis\FSCN\data\abide_val.npy
test_data_path: W:\Brain Analysis\FSCN\data\abide_test.npy
```

或者在命令行中指定：

```bash
python train.py --use_split_data --train_data_path data/abide_train.npy --val_data_path data/abide_val.npy --test_data_path data/abide_test.npy
```

## 步骤 3: 运行训练

使用配置文件运行训练：

```bash
python train.py --config args/abide.yml
```

或者使用命令行参数：

```bash
python train.py --use_split_data
```

## 训练输出

训练过程中会显示：
1. 训练集和验证集的性能指标（每个 epoch）
2. 最佳验证集性能
3. 测试集上的最终评估结果，包括：
   - Loss
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Sensitivity (TPR)
   - Specificity (TNR)
   - AUROC
   - 混淆矩阵
   - 分类报告

## 两种使用模式

### 模式 1: 使用预分割数据（推荐）

```yaml
use_split_data: true
```

优点：
- 数据分割固定，实验可重复
- 有独立的测试集用于最终评估
- 符合标准的机器学习实践

### 模式 2: 动态分割数据

```yaml
use_split_data: false
```

此模式下：
- 数据在运行时按照 `val_split` 比例分割
- 只有训练集和验证集，没有测试集
- 适合快速实验

## 代码改动说明

### 新增文件
- `split_data.py`: 数据分割脚本

### 修改文件

#### data_utils.py
- 新增 `load_abide_split_data()`: 加载已分割的数据文件
- 新增 `prepare_task_data_from_dict()`: 从预分割的数据字典准备数据
- 修改 `print_task_summary()`: 支持显示测试集信息

#### train.py
- 修改 `train_single_task()`: 添加测试集参数和评估
- 修改 `parse_args()`: 添加数据分割相关参数
- 修改 `main()`: 支持加载预分割数据和测试集评估

#### args/abide.yml
- 添加 `use_split_data` 配置项
- 添加训练集、验证集、测试集的路径配置

## 注意事项

1. **首次使用**: 必须先运行 `split_data.py` 生成分割后的数据文件
2. **随机种子**: 数据分割使用 `random_state=42` 确保可重复性
3. **存储空间**: 分割后的三个文件总大小与原文件相同
4. **数据完整性**: 分割过程会保留所有数据字段（corr, pcorr, label, site, timeseries）

## 示例输出

训练完成后会看到类似以下的输出：

```
================================================================================
Test Set Performance:
================================================================================
  Loss: 0.4523
  Accuracy: 0.7850
  Precision: 0.7654
  Recall: 0.8123
  F1-Score: 0.7882
  Sensitivity: 0.8123
  Specificity: 0.7584
  AUROC: 0.8456
================================================================================
```

## 问题排查

如果遇到文件找不到的错误：
1. 确认已运行 `split_data.py`
2. 检查配置文件中的路径是否正确
3. 确认数据文件在正确的目录下

如果想回到原来的单文件模式：
- 设置 `use_split_data: false` 或
- 在命令行中省略 `--use_split_data` 参数

