class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y
        return out

    def backward(self,dout):
        dx = dout*self.y
        dy = dout*self.x # 상류에서 넘어온 미분(dout)에 순전파 떄의 값을 서로 바꿔 곱한뒤 하류로 흘린다.

        return dx, dy

apple = 100
apple_num = 2
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(apple_price) # 200
print(price) # 220

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple)
print(dapple_num)
print(dtax)