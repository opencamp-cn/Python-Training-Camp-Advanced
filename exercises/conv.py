import numpy as np
def conv2d(input_img, kernel):
    h, w = input_img.shape
    kH, kW = kernel.shape
    output_h = h - kH + 1
    output_w = w - kW + 1
    if output_h <= 0 or output_w <= 0:
        return np.array([])
        
    output = np.zeros((output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            window = input_img[i:i+kH, j:j+kW]
            output[i, j] = np.sum(window * kernel)
    return output
