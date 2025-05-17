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
        
        # 检测轮廓 (OpenCV 4.x版本)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建副本用于绘制
        img_contour = img.copy()
        
        # 绘制轮廓
        cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 2)
        
        return img_contour, [cnt.tolist() for cnt in contours]
    except Exception as e:
        print(f"Error in contour detection: {e}")
        return None, None
