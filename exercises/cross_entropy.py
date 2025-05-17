# exercises/cross_entropy.py
"""
练习：交叉熵损失 (Cross Entropy Loss)

描述：
实现分类问题中常用的交叉熵损失函数。

请补全下面的函数 `cross_entropy_loss`。
"""
import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    计算交叉熵损失。

    Args:
        y_true (np.array): 真实标签 (独热编码或类别索引)。
                           如果 y_true 是类别索引, 它将被转换为独热编码。
                           形状: (N,) 或 (N, C)，N 是样本数, C 是类别数。
        y_pred (np.array): 模型预测概率，形状 (N, C)。
                           每个元素范围在 [0, 1]，每行的和应接近 1。

    Return:
        float: 平均交叉熵损失。
    """
    # 获取类别数量
    C = y_pred.shape[1]
    
    # 如果y_true是类别索引，转换为独热编码
    if y_true.ndim == 1:
        y_true = np.eye(C)[y_true]
    
    # 数值稳定性处理
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    
    # 计算交叉熵损失
    loss = -np.sum(y_true * np.log(y_pred))
    
    # 计算平均损失
    return loss / y_true.shape[0]
