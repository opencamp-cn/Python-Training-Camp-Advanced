# exercises/image_processing.py
"""
练习：图像基本处理

描述：
使用 OpenCV 实现基本的图像读取、灰度转换、高斯滤波和边缘检测。

请补全下面的函数 `image_processing_pipeline`。
"""
import os
import traceback
from inspect import trace
from pathlib import Path

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
    # 请在此处编写代码
    # 提示：
    # 1. 使用 cv2.imread() 读取图像。
    # 2. 检查图像是否成功读取（img is None?）。
    # 3. 使用 cv2.cvtColor() 将图像转为灰度图 (cv2.COLOR_BGR2GRAY)。
    # 4. 使用 cv2.GaussianBlur() 进行高斯滤波。
    # 5. 使用 cv2.Canny() 进行边缘检测。
    # 6. 使用 try...except 包裹代码以处理可能的异常。
    try:
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(Path(current_dir).parent, image_path)
        file_path = Path(file_path).as_posix()
        img = cv2.imread(file_path)
        if img is None:
            raise Exception(f"read image from {file_path} failed")
        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        blur_image = cv2.GaussianBlur(v, (5,5), 0)
        edges = cv2.Canny(blur_image, 100, 150)
        return edges
    except Exception as e:
        traceback.print_exc()
        return None
