# exercises/iou.py
"""
练习：计算交并比 (Intersection over Union, IoU)

描述：
实现用于计算两个边界框之间交并比 (IoU) 的函数。
IoU 是目标检测任务中常用的评估指标。

请补全下面的函数 `calculate_iou`。
"""
import numpy as np

def calculate_iou(box1, box2):
    """
    计算两个边界框 (bounding box) 的交并比 (IoU)。
    边界框格式：[x_min, y_min, x_max, y_max]

    Args:
        box1 (list or np.array): 第一个边界框的坐标 [x1_min, y1_min, x1_max, y1_max]。
        box2 (list or np.array): 第二个边界框的坐标 [x2_min, y2_min, x2_max, y2_max]。

    Return:
        float: 计算得到的 IoU 值，范围在 [0, 1]。
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

    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area

    # 计算IoU，处理除零情况
    return intersection_area / union_area if union_area > 0 else 0.0
