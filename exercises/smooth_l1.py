import numpy as np
# 重命名函数并修改参数为 x 和 sigma
def smooth_l1(x, sigma=1.0):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    
    # 根据 sigma 计算 beta (对应原实现中的 beta = 1 / sigma**2)
    sigma2 = sigma**2
    beta = 1.0 / sigma2
    
    abs_x = np.abs(x)
    # 使用与 docstring/测试用例一致的条件 |x| < 1 / sigma**2 (即 abs_x < beta)
    condition = abs_x < beta 
    
    # 计算损失，匹配 docstring/测试用例的公式
    l2_loss = 0.5 * (sigma * x)**2 
    l1_loss = abs_x - 0.5 / sigma2 # 等价于 abs_x - 0.5 * beta
    
    elementwise_loss = np.where(condition, l2_loss, l1_loss)
    # 注意：测试用例计算的是 elementwise loss 的均值，但函数本身应返回 elementwise loss
    # loss = np.mean(elementwise_loss) # 不在这里计算均值

    return elementwise_loss # 返回每个元素的损失值
