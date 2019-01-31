def SVM_Loss(y, t): # t는  라벨
    a = 0
    for i in range(len(y)):
        b = max(0, y[i]-y[t]+1)
        a += b
    return a - 1

y1 = [3.2, 5.1, -1.7]
y2 =  [1.3, 4.9, 2.0]
y3 = [2.2, 2.5, -3.1]

t0 = 0
t1 = 1
t2 = 2


print(SVM_Loss(y1,t0)) # 2.89
print(SVM_Loss(y2,t1)) # 0
print(SVM_Loss(y3,t2)) # 12.9