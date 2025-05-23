# exercises/conv.py
"""
练习：二维卷积 (Convolution)

描述：
实现一个简单的二维卷积操作。

请补全下面的函数 `conv2d`。
"""
import numpy as np

def conv2d(x, kernel):
    """
    执行二维卷积操作 (无填充, 步幅为 1)。

    Args:
        x (np.array): 输入二维数组, 形状 (H, W)。
        kernel (np.array): 卷积核二维数组, 形状 (kH, kW)。

    Return:
        np.array: 卷积结果, 形状 (out_H, out_W)。
                  out_H = H - kH + 1
                  out_W = W - kW + 1
    """
    # 获取输入和卷积核的形状
    H, W = x.shape
    kH, kW = kernel.shape
    
    # 计算输出形状
    out_H = H - kH + 1
    out_W = W - kW + 1
    
    # 初始化输出数组
    out = np.zeros((out_H, out_W))
    
    # 遍历输出数组的每个位置
    for i in range(out_H):
        for j in range(out_W):
            # 提取输入中与当前卷积核对应的区域
            patch = x[i:i+kH, j:j+kW]
            # 计算点乘并求和
            out[i, j] = np.sum(patch * kernel)
    
    return out
