import numpy as np
# x0와 x1의 편미분을 변수별로 따로 계산
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val # f(x-h) 계산
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원

    return grad

def function_2(x):
    return x[0]**2 + x[1]**2
    # 또는 return np.sum(x**2)

print(numerical_gradient(function_2, np.array([3.0,4.0]))) # 3.00005, 4.00005
print(numerical_gradient(function_2, np.array([0.0,2.0]))) # 4.99999-e5 2.000005