# exercises/maxpool.py
"""
练习：最大池化 (Max Pooling)

描述：
实现一个简单的二维最大池化操作。

请补全下面的函数 `maxpool`。
"""
import numpy as np

def maxpool(x, kernel_size, stride):
    """
    执行二维最大池化操作。

    Args:
        x (np.array): 输入二维数组，形状 (H, W)。
        kernel_size (int): 池化窗口的大小 (假设为正方形 k x k)。
        stride (int): 池化窗口移动的步幅。

    Return:
        np.array: 最大池化结果，形状 (out_H, out_W)。
                  out_H = (H - kernel_size) // stride + 1
                  out_W = (W - kernel_size) // stride + 1
    """
    # 获取输入尺寸
    H, W = x.shape
    
    # 计算输出尺寸
    out_H = (H - kernel_size) // stride + 1
    out_W = (W - kernel_size) // stride + 1
    
    # 初始化输出数组
    out = np.zeros((out_H, out_W))
    
    # 遍历输出数组的每个位置
    for i in range(out_H):
        for j in range(out_W):
            # 计算当前窗口在输入中的起始位置
            h_start = i * stride
            w_start = j * stride
            
            # 提取当前窗口
            window = x[h_start:h_start+kernel_size, w_start:w_start+kernel_size]
            
            # 计算窗口中的最大值并存入输出
            out[i, j] = np.max(window)
    
    return out