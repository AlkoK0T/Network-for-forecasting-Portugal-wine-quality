from network import network
from minmax import minmax
import numpy as np

hiddennodes = [11,20,40,20,11]
tdf = open("winequality-red.csv", 'r')
tdl = tdf.readlines()
tdl = tdl[1:]
tdf.close()
networkred=network(hiddennodes,0.1)
networkred.epoch(tdl,5)
#region result
tedf = open("winequality-red.csv", 'r')
tedl = tedf.readlines()
tedl = tedl[1:]
tedf.close()
pr=0
scorecard=[]
minm=minmax()
for rec in tedl:
    all_val=[j for j in rec.split(';')]
    targets = np.zeros(11)
    targets[int(all_val[-1])+1]=1
    chngevar=[minm.coefficient(j,float(all_val[j])) for j in range(len(all_val)-1) ]
    inputs = (np.asarray(chngevar,dtype=float))
    out=networkred.query(inputs)
    if np.argmax(out)==int(all_val[-1]):
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass
scorecard_array = np.asarray(scorecard)
print ("эффективность = ", scorecard_array.sum() / scorecard_array.size)
#endregion