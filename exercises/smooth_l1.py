# exercises/smooth_l1.py
"""
练习：Smooth L1 损失函数

描述：
实现 Smooth L1 损失函数，常用于目标检测中的边界框回归。

请补全下面的函数 `smooth_l1`。
"""
import numpy as np

def smooth_l1(x, sigma=1.0):
    """
    计算 Smooth L1 损失。
    公式:
        0.5 * (sigma * x)**2   if |x| < 1 / sigma**2
        |x| - 0.5 / sigma**2   otherwise

    Args:
        x (np.array): 输入差值数组，任意形状。
        sigma (float): 控制平滑区域的参数，默认为 1.0。

    Return:
        np.array: 计算得到的 Smooth L1 损失数组，形状与输入相同。
    """
    sigma2 = sigma ** 2
    abs_x = np.abs(x)
    condition = abs_x < 1.0 / sigma2
    
    # 计算两部分损失
    quadratic = 0.5 * sigma2 * x ** 2
    linear = abs_x - 0.5 / sigma2
    
    # 根据条件选择损失值
    return np.where(condition, quadratic, linear)
