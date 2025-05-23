# exercises/image_processing.py
import cv2
import numpy as np

def image_processing_pipeline(image_path):
    """
    使用 OpenCV 读取图像，进行高斯滤波和边缘检测。
    参数:
        image_path: 图像文件的路径 (字符串).
    返回:
        edges: Canny 边缘检测的结果 (NumPy 数组, 灰度图像).
               如果读取图像失败, 返回 None.
    """
    try:
        # 1. 使用 OpenCV 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None

        # 2. 将图像转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. 应用高斯滤波进行降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4. 使用 Canny 边缘检测
        edges = cv2.Canny(blurred, 50, 150)

        return edges

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
