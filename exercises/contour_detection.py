# exercises/contour_detection.py
"""
练习：轮廓检测

描述：
使用 OpenCV 检测图像中的轮廓并将其绘制出来。

请补全下面的函数 `contour_detection`。
"""
import cv2
import numpy as np

def contour_detection(image_path):
    """
    使用 OpenCV 检测图像中的轮廓
    参数:
        image_path: 图像路径
    返回:
        tuple: (绘制轮廓的图像, 轮廓列表) 或 (None, None) 失败时
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        
        # 转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 检测轮廓（处理不同OpenCV版本的返回值差异）
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 将轮廓转换为列表（如果还不是列表）
        contour_list = list(contours) if not isinstance(contours, list) else contours
        
        # 创建图像副本用于绘制
        result_img = img.copy()
        
        # 绘制轮廓
        cv2.drawContours(result_img, contour_list, -1, (0, 255, 0), 2)
        
        return result_img, contour_list
    
    except Exception as e:
        return None, None