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
    # 请在此处编写代码
    # (与 iou.py 中的练习相同，可以复用代码或导入)
    # 提示：计算交集面积和并集面积，然后相除。
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box1_area=(box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area=(box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - intersection_area
    IoU = 0 if union_area==0 else intersection_area / union_area
    return IoU

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
    # 请在此处编写代码
    # 提示：
    # 1. 如果 boxes 为空，直接返回空列表。
    # 2. 将 boxes 和 scores 转换为 NumPy 数组。
    # 3. 计算所有边界框的面积 areas。
    # 4. 根据 scores 对边界框索引进行降序排序 (order = np.argsort(scores)[::-1])。
    # 5. 初始化一个空列表 keep 用于存储保留的索引。
    # 6. 当 order 列表不为空时循环：
    #    a. 取出 order 中的第一个索引 i (当前分数最高的框)，加入 keep。
    #    b. 计算框 i 与 order 中剩余所有框的 IoU。
    #       (需要计算交集区域坐标 xx1, yy1, xx2, yy2 和交集面积 intersection)
    #    c. 找到 IoU 小于等于 iou_threshold 的索引 inds。
    #    d. 更新 order，只保留那些 IoU <= threshold 的框的索引 (order = order[inds + 1])。
    # 7. 返回 keep 列表。
    if len(boxes)<=0:
        return boxes
    areas=[]
    for box1 in boxes:
        areas.append((box1[2]-box1[0])*(box1[3]-box1[1]))
    boxes=np.array(boxes)
    scores=np.array(scores)
    order = np.argsort(scores)[::-1].tolist()
    # print(order)
    i=order.pop(0)
    keep=[i]
    for v in range(len(order)):
        # print(calculate_iou(boxes[i],boxes[order[v]]))
        if calculate_iou(boxes[i],boxes[order[v]])<=iou_threshold:
            keep.append(order[v])
    return keep

if __name__=='__main__':
    boxes2 = [
        [10, 10, 60, 60],    # 得分最高
        [15, 15, 65, 65],    # 与第一个框IoU≈0.53
        [50, 50, 100, 100],  # 与第一个框IoU≈0.11
        [150, 150, 200, 200] # 完全独立
    ]
    scores2 = [0.9, 0.85, 0.8, 0.75]
    result2 = nms(boxes2, scores2, 0.5)
    print(result2)
