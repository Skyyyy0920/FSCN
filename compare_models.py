import scipy.io
import numpy as np

# 读取 .mat 文件
mat_data = scipy.io.loadmat('data/PPMI/fcnHCPD.mat')

# 获取 fcn_corr 数据
fcn_corr = mat_data['fcn_corr']

print("外层数组形状:", fcn_corr.shape)
print("外层数组类型:", fcn_corr.dtype)

# 查看第一个元素
print("\n第一个元素的信息:")
first_element = fcn_corr[0, 0]
print("类型:", type(first_element))
print("形状:", first_element.shape)
print("数据类型:", first_element.dtype)

# 查看几个样本的形状
print("\n前5个元素的形状:")
for i in range(min(5, fcn_corr.shape[0])):
    print(f"元素 {i}: {fcn_corr[i, 0].shape}")

# 如果想访问所有88个受试者的数据
print(f"\n总共有 {fcn_corr.shape[0]} 个样本")

# 提取第一个样本的数据作为示例
sample_data = fcn_corr[0, 0]
print("\n第一个样本的统计信息:")
print("最小值:", np.min(sample_data))
print("最大值:", np.max(sample_data))
print("平均值:", np.mean(sample_data))
print("标准差:", np.std(sample_data))

# 查看前几个值
print("\n数据预览:")
print(sample_data[:5, :5] if sample_data.ndim == 2 else sample_data[:10])