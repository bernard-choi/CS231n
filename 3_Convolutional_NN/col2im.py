import numpy as np
def col2im(col, input_shape, filter_h, filter_w, stride = 1, pad = 0):
    """(im2col과는 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    :param col:  2차원 배열
    :param input_shape: 원래 이미지 데이터의 형상 ex(10,1,28,28)
    :param filter_h:  필터의 높이
    :param filter_w:  필터의 너비
    :param stride:  스트라이드
    :param pad:  패딩
    :return:
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]