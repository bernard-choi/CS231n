import numpy as np
def im2col(input_data, filter_h, filter_w, stride = 1, pad = 0):
    """

    :param input_data: 4차원 배열 형태의 입력 데이터( 이미지수, 채널 수 , 높이, 너비)
    :param filter_h: 필터의 높이
    :param filter_w: 필터의 너비
    :param stride: 스트라이드
    :param pad: 패딩
    :return:
    col : 2차원 배열
    """

    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w,-1)
    return col

# im2col 예시
x1 = np.random.rand(1,3,7,7)   # (데이터 수, 채널 수, 높이, 너비)
col1 = im2col(x1,5,5,stride=1,pad=0)
print(col1.shape)              # (9, 75)
                               # stride 1 이므로 가로3 세로3 9개의 헹, 한 행에 5X5X3개의 parameter

# im2col 예시2
x2 = np.random.rand(10,3,7,7,) # (데이터 수, 채널 수, 높이, 너비)
col2 = im2col(x2, 5, 5, stride=1, pad = 0)
print(col2.shape)              # (90, 75)
                               # 데이터 수 X 3 X 3 = 90
                               # 5 X 5 X 3 = 75


