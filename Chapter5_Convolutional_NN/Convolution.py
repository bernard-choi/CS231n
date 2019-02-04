import numpy as np
from . import im2col
from . import col2im

class Convolution:
    def __init__(self,w,b,stride=1,pad=0):
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad

        # 중간 데이터(backward시 사용)
        self.x = None
        self.col = None
        self.col_w = None

        # 가중치와 편향 매개변수의 기울기
        self.dw = None
        self.db = None

    def forward(self,x):
        FN, C, FH, FW = self.w.shape # FN:필터개수, C:채널수, FH:필터높이, FW: 필터넓이
        N, C, H, W = x.shape # N개의 데이터
        out_h = int(1+(H+2*self.pad - FH) / self.stride)
        out_w = int(1+(W + 2*self.pad - FW) / self.stride)

        col = im2col(x,FH,FW,self.stride, self.pad)
        col_W = self.w.reshape(FN, -1).T # 필터전개
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)  # (N, H, W, C) -> (N, C, H, W)

        return out

     def backward(self,dout):
         FN, C, FH, FW = self.w.shape
         dout = dout.transpose(0,2,3,1).reshape(-1, FN)

         self.db = np.sum(dout, axis =0)
         self.dw = np.dot(self.col.T, dout)
         self.dw = self.dw.transpose(1,0).reshape(FN, C, FH, FW)

         dcol = np.dot(dout, self.col_w.T)
         dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

         return dx

x = np.random.rand(1,3,32,32)
w = np.random.rand(6,3,5,5) # 필터수 6개, 5X5
b = np.random.rand(784,6) # 28*28 = 784

conv = Convolution(w,b)
print(conv.forward(x).shape) # (1, 6, 28, 28)