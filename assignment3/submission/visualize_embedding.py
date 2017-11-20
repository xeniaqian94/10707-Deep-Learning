import pickle
import sys

from train import Vocab, Model
from util import *

label = sys.argv[1]  # -lr_1.0_-max_epoch_200

sequences=[['use','a','great','tool'],['for','extra','points','you'],['language','generation','task','is'],['the','united','states','of'],['gone','with','the','wind']]

vocab = pickle.load(open("dump/vocab_" + label, "r"))
i2w=dict()
for key in vocab.w2i.keys():
    i2w[vocab.w2i[key]]=key

# model = pickle.load(open("dump/model_" + label, "r"))
# embedding=model.C[:,:2]
# print embedding.shape





embedding=pickle.load(open("dump/embedding_pytorch_rnn__embed_size_2","r"))
print embedding.shape





import matplotlib.pyplot as plt
# plt.plot([0.1,0.2,0.3,0.4], [-0.1,-0.4,0.9,0.16], 'ro')
# plt.axis([-1, 1, -1, 1])
# plt.show()

x=[]
y=[]
label=[]

for i in [int(this) for this in np.random.rand(500)*6000]:
# for i in range(100,600):
# for i in range(500):
    x+=[embedding[i,0]]
    y+=[embedding[i,1]]
    label+=[i2w[i]]


fig = plt.figure()
ax = fig.add_subplot(111)

# A = -0.75, -0.25, 0, 0.25, 0.5, 0.75, 1.0
# B = 0.73, 0.97, 1.0, 0.97, 0.88, 0.73, 0.54
# label='ha','ee','we','sd','wr','cv','sg'

plt.plot(x,y,'ro')
for ind,xy in enumerate(zip(x, y)):
    ax.annotate(label[ind], xy=(xy[0],xy[1]), textcoords='data')

plt.grid()

plt.axis([np.percentile(x, 9, axis=0), np.percentile(x, 91, axis=0), np.percentile(y,9, axis=0), np.percentile(y, 91, axis=0)])
plt.show()



