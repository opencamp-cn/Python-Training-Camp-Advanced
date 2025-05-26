# exercises/contour_detection.py
"""
练习：轮廓检测

描述：
使用 OpenCV 检测图像中的轮廓并将其绘制出来。

请补全下面的函数 `contour_detection`。
"""
import os
import traceback
from pathlib import Path

import cv2
import numpy as np


def convert_contours_to_lists(contours):
    """将 NumPy 数组形式的轮廓转换为 Python 列表"""
    contours_list = []
    for contour in contours:
        # 将单个轮廓从NumPy数组转换为列表
        contour_list = contour.tolist()
        contours_list.append(contour_list)
    return contours_list

def contour_detection(image_path):
    """
    使用 OpenCV 检测图像中的轮廓
    参数:
        image_path: 图像路径
    返回:
        tuple: (绘制轮廓的图像, 轮廓列表) 或 (None, None) 失败时
    """
    # 请在此处编写代码
    # 提示：
    # 1. 使用 cv2.imread() 读取图像。
    # 2. 检查图像是否成功读取。
    # 3. 使用 cv2.cvtColor() 转为灰度图。
    # 4. 使用 cv2.threshold() 进行二值化处理。
    # 5. 使用 cv2.findContours() 检测轮廓 (注意不同 OpenCV 版本的返回值)。
    # 6. 创建图像副本 img.copy() 用于绘制。
    # 7. 使用 cv2.drawContours() 在副本上绘制轮廓。
    # 8. 返回绘制后的图像和轮廓列表。
    # 9. 使用 try...except 处理异常。
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(Path(current_dir).parent, image_path)
    file_path = Path(file_path).as_posix()
    # raise Exception(f"{file_path}")
    try:
        if not os.path.exists(file_path):
            raise Exception(f"{file_path} not exists")
        img = cv2.imread(file_path)

        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        # cv2.imshow("gray", image_hsv)
        ret, mask = cv2.threshold(v, 127, 255, cv2.THRESH_BINARY)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # 提取轮廓，重要的是contours这个数组类型
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Wait for a key press and then terminate all OpenCV windows
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img_contours = img.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 1)
        # cv2.imshow("copy with contours", img_contours)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        contours_list = convert_contours_to_lists(contours)
        return img_contours, contours_list
    except:
        traceback.print_exc()
        return None, None
