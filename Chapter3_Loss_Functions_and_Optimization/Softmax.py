import numpy as np

def softmax(a): # 지수함수를 계산할 때 값이 커지면 nan이 계산됨
    c = np.max(a)
    exp_a = np.exp(a-c) # 소프트 맫그의 지수함수를 계산할 떄 어떤 정수를 더해도 결과는 바뀌지 않는다는 점을 이용
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

a = [3.2, 5.1, -1.7]

print(softmax(a))