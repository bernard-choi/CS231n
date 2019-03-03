import numpy as np

def cross_entropy_error(y, t):                       # t는 라벨
    delta = 1e-7
    return -np.sum(t* np.log(y + delta))

# 예시
t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t))) # 0.5108254

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t))) # 틀린 경우 늘어남 2.302584