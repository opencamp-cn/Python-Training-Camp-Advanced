# exercises/nms.py
"""
练习：非极大值抑制 (Non-Maximum Suppression, NMS)

描述：
实现目标检测中常用的 NMS 算法，用于去除重叠度高的冗余边界框。

请补全下面的函数 `calculate_iou` 和 `nms`。
"""
import numpy as np

def calculate_iou(box1, box2):
    """
    计算两个边界框的交并比 (IoU)。
    边界框格式：[x_min, y_min, x_max, y_max]

    Args:
        box1 (np.array): 第一个边界框 [x1_min, y1_min, x1_max, y1_max]。
        box2 (np.array): 第二个边界框 [x2_min, y2_min, x2_max, y2_max]。

    Return:
        float: IoU 值。
    """
    # 计算相交区域坐标
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # 计算相交区域面积
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    
    # 计算各自面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积和IoU
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou

def nms(boxes, scores, iou_threshold):
    """
    执行非极大值抑制 (NMS)。

    Args:
        boxes (np.array): 边界框数组，形状 (N, 4)，格式 [x_min, y_min, x_max, y_max]。
        scores (np.array): 每个边界框对应的置信度分数，形状 (N,)。
        iou_threshold (float): IoU 阈值，用于判断是否抑制。

    Return:
        list: 保留下来（未被抑制）的边界框的索引列表。
    """
    # 处理空输入
    if len(boxes) == 0:
        return []
    
    # 转换为NumPy数组
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 根据分数降序排序
    order = np.argsort(scores)[::-1]
    
    keep = []
    while order.size > 0:
        # 取出当前最高分的框
        i = order[0]
        keep.append(i)
        
        # 计算与剩余框的IoU
        ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in order[1:]])
        
        # 找到IoU小于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        
        # 更新order，保留符合条件的框
        order = order[inds + 1]
    
    return keep