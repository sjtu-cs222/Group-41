import pandas as pd
import numpy as np

f = open("training loss.txt")

lossLst = []
tmp = []

for line in f.readlines():
    line.strip()
    lst = line.split(' ')
    try:
        if lst[5] == '':
            tmp.append(lst[4])
        else:
            tmp.append(lst[5])
    except:
        lossLst.append(tmp)
        tmp = []
        continue
lossLst.append(tmp)
print(lossLst[0])
print(lossLst[6])
print(lossLst[-1])
res =[lossLst[0], lossLst[6], lossLst[-1]]
res = np.array(res, ndmin = 2)
res = np.transpose(res)
pre_save = pd.DataFrame(columns = ['0', '1','2'], data = res)
pre_save.to_csv("EvaluationLoss.csv")