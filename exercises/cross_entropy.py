import numpy as np
def cross_entropy_loss(y_true, y_pred):
    # 处理 y_true 可能为类别索引的情况
    N = y_pred.shape[0]
    C = y_pred.shape[1]
    if y_true.ndim == 1:
        # 将类别索引转换为独热编码
        y_true_one_hot = np.zeros((N, C))
        y_true_one_hot[np.arange(N), y_true] = 1
        y_true = y_true_one_hot
        
    # 防止 log(0) 错误
    epsilon = 1e-12
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0)
    
    # 计算交叉熵损失
    # 使用修正后的 y_true (独热编码)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / N
    
    return loss
