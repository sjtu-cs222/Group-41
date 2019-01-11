import numpy as np
import pandas as pd

baseline=[0.9597, 0.9968, 0.9665, 0.9788, 0.9569, 0.9918, 0.9236, 0.9872, 0.9942, 0.9915]
correct=[]
for i in range(10):
    t = []
    correct.append(t)

RESERVED_RATIO = 0.25

f = open("Evaluation.csv")
df = pd.read_csv(f)
rawInput = df.iloc[:, 1:].values
rawInput = np.transpose(rawInput)
for i in range(10):
    correct[i] = list(rawInput[i])

cutIndex = int(len(correct[i]) * RESERVED_RATIO)
cutValue = []

for i in range(10):
    tmp = correct[i][:]
    tmp.sort()
    cutValue.append(tmp[cutIndex - 1])

reservedMatrix = []
for i in range(10):
    reservedMatrix.append([0] * 64)
    c = 0
    t = len(correct[i]) - 1
    while c <= 16 and t >= 0:
        if correct[i][t] <= cutValue[i]:
            reservedMatrix[i][t] = 1
            c = c + 1
        t = t - 1
#reservedMatrix = np.array(reservedMatrix)
#print(reservedMatrix)
for i in range(64):
    print(reservedMatrix[1][i] + reservedMatrix[5][i] + reservedMatrix[6][i], end = " ")