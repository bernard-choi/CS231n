import numpy as np
import matplotlib.pyplot as plt


def numerical_diff(f,x):
    h = 1e-4 # 너무 작은 숫자 넣으면 계산 X
    return (f(x+h) - f(x-h)) / (2*h)

# 예시함수
def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0,20,0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("f(x) = 0.01*x^2 + 0.1*x")
plt.plot(x,y)
plt.show()

print(numerical_diff(function_1,5))
print(numerical_diff(function_1,10))