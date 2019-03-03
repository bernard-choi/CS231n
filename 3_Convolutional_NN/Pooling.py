import numpy as np
from . import im2col
from . import col2im

class Pooling:
    def __init__(self, pool_h, pool_w, stride = 1, pad = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self,x):
        N, C, H, W = x.shape # (batch_size, colour, height, weight)
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 전개 (1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 최대값(2)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        return out

    def backward(self, x):
        dout = dout.transpose(0,2,3,1)

        pool_size = self.pool_h*self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout,shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


x = np.array([[[[1,1,2,4],
                [5,6,7,8],
                [3,2,1,0],
                [1,2,3,4]]]])

Pool = Pooling(2,2,stride = 2)
print(Pool.forward(x))

# array([[[[6, 8],
#          [3, 4]]]])
