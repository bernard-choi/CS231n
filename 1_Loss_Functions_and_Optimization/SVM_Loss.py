def SVM_Loss(scores, y, safety_margin=1):
    """
    scores : w.dot(x)
    y : 라벨
    safety_margin : safety_margin
    """
    margins = np.maximum(0,scores - scores[y] + safety_margin)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i


scores1 = np.array([3.2, 5.1, -1.7])
scores2 =  np.array([1.3, 4.9, 2.0])
scores3 = np.array([2.2, 2.5, -3.1])

y0 = 0
y1 = 1
y2 = 2


print(SVM_Loss(scores1,y0)) # 2.89
print(SVM_Loss(scores2,y1)) # 0
print(SVM_Loss(scores3,y2)) # 12.9