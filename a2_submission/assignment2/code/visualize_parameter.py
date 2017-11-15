import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from train import Model

model=pickle.load(open(sys.argv[1],"r"))

print len(model.W)
print model.W[1].shape

fig=plt.figure()
f, axarr = plt.subplots(int(np.sqrt(model.W[1].shape[0])), int(np.sqrt(model.W[1].shape[0])))
plt.gray()

print model.W[1].shape[0]
for i in np.arange(int(model.W[1].shape[0])):
    row=i/10
    col=i%10
    this_array=np.asarray(model.W[1][i]).reshape(np.sqrt(len(model.W[1][i])),np.sqrt(len(model.W[1][i])))
    axarr[row,col].imshow(this_array)
    axarr[row, col].axis('off')


f.subplots_adjust()
plt.show()
fig.savefig("../plot/"+sys.argv[1].split("/")[-1]+".png")

plt.cla()





