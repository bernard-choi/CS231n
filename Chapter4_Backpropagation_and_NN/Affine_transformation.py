import numpy as np
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0) # 순전파 때의 편향 덧셈은 XW에 대한 편향이 각 데이터에 더해집니다. 예를 들어 N=2로 한 경우 편향은 그 두 데이터 각각에 더해진다.
                                       # 그래서 역전파 때는 각 데이터의 역전파 값이 편향의 원소에 모여야 한다.
        return dx