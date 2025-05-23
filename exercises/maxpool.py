import numpy as np
def maxpool(input_map, kernel_size, stride):
    input_h, input_w = input_map.shape
    pool_h, pool_w = kernel_size, kernel_size
    stride_h, stride_w = stride, stride

    output_h = (input_h - pool_h) // stride_h + 1
    output_w = (input_w - pool_w) // stride_w + 1

    output_map = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            start_row = i * stride_h
            start_col = j * stride_w
            window = input_map[start_row:start_row+pool_h, start_col:start_col+pool_w]
            max_val = np.max(window)
            output_map[i, j] = max_val

    return output_map
