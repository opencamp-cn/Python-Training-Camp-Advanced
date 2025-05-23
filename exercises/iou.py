import numpy as np
def calculate_intersection(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0
    
    intersection = (x2 - x1) * (y2 - y1)
    return intersection

def calculate_area(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return area

def calculate_iou(box1, box2):
    intersection = calculate_intersection(box1, box2)
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    iou = intersection / union
    return iou
