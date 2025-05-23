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
        # 1. 读取图像
        img = cv2.imread(image_path)
        if img is None:
            return None, None

        # 2. 灰度转换
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. 二值化
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 4. 轮廓检测（OpenCV不同版本返回值不同）
        contours, _ = cv2.findContours(
            thresh, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 将NumPy数组的元组转换为Python列表
        contours = list(contours) if contours is not None else []

        # 5. 绘制轮廓
        result = img.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        return result, contours

    except Exception:
        return None, None
