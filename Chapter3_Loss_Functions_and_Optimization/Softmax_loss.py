import numpy as np

def cross_entropy_error(y, t): # t는 라벨
    delta = 1e-7
    return -np.sum(t* np.log(y + delta))

def softmax(a): # 지수함수를 계산할 때 값이 커지면 nan이 계산됨
    c = np.max(a)
    exp_a = np.exp(a-c) # 소프트 맫그의 지수함수를 계산할 떄 어떤 정수를 더해도 결과는 바뀌지 않는다는 점을 이용
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

X = [3.2, 5.1, -1.7] # 라벨이 1이면 loss가 최소일 것이다.
t1 = [1,0,0]
t2 = [0,1,0]
t3 = [0,0,1]

print(softmax_loss(X,t1)) # 2.04035 피피티는 자연로그가 아니라 값이 다르다. np.log는 자연로그
print(softmax_loss(X,t2)) # 0.14035
print(softmax_loss(X,t3)) # 6.94025


