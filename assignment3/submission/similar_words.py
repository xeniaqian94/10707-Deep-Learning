import pickle
import sys
from sklearn import preprocessing

from train import Vocab, Model
from util import *

label = sys.argv[1]  # -lr_1.0_-max_epoch_200


model = pickle.load(open("dump/model_" + label, "r"))
vocab = pickle.load(open("dump/vocab_" + label, "r"))


print model.C.shape

print model.C[200:205,:]
raw_input()


i2w=dict()

for key in vocab.w2i.keys():
    i2w[vocab.w2i[key]]=key

print type(model.C[0])


distance=dict()

top=1500

#
# for i in range(model.C.shape[0]-1):
#     for j in range(i+1,model.C.shape[0]):

pool=range(500)+range(4250,4750)+range(7000,7500)

for i in range(len(pool)-1):
    print i
    for j in range(i+1,len(pool)):
        if np.random.rand()>0.5:
            distance[(i,j)]=np.linalg.norm(model.C[pool[i]]-model.C[pool[j]])

print "sorting"
sorted_distance = sorted(distance.items(), key=operator.itemgetter(1))

print "sorting finished"

# sorted_distance=pickle.load(open("sorted_distance","r"))
pickle.dump(sorted_distance,open("sorted_distance","w"))

print "most similar: "
list1=[]
for i in range(100):
    list1+=[((i2w[sorted_distance[i][0][0]],  i2w[sorted_distance[i][0][1]]),  sorted_distance[i][1])]
print list1

print "least similar"
list2=[]
for i in range(len(sorted_distance)-400,len(sorted_distance)-300):
    list2+=[(i2w[sorted_distance[i][0][0]], i2w[sorted_distance[i][0][1]]), sorted_distance[i][1]]


print list2

print "Pairs of selected similar words"
# words=[("cool","cold"),("great","good"),("good","bad"),("west","east"),("fall","winter"),("fall","summer"),("car","automobile"),]


words=[("west","east"),("fall","summer"),("spring","summer"),("car","automobile"),("left","right"),("fall","winter"),("good","bad"),("states","john"),("considers","securities"),("benefits","john"),("applications","why")]
for pair in words:
    # X_normalized = preprocessing.normalize(model.C[vocab.word_to_id(pair[0])], norm='l2')
    # Y_normalized=preprocessing.normalize(model.C[vocab.word_to_id(pair[1])], norm='l2')
    # print pair,np.linalg.norm(X_normalized-Y_normalized)
    # print pair,vocab.word_to_id(pair[0]),vocab.word_to_id(pair[1]), np.linalg.norm(model.C[vocab.wo
    # rd_to_id(pair[0])]-model.C[vocab.word_to_id(pair[1])])
    print pair, np.linalg.norm(model.C[vocab.word_to_id(pair[0])]-model.C[vocab.word_to_id(pair[1])])

# print
# print "Pairs of selected dissimilar words"
# words=[("good","west"),("fall","car"),("summer","automobile"),("good","east"),("winter","car"),("united","john"),("states","john"),]
#
# for pair in words:
#     # X_normalized = preprocessing.normalize(model.C[vocab.word_to_id(pair[0])], norm='l2')
#     # Y_normalized = preprocessing.normalize(model.C[vocab.word_to_id(pair[1])], norm='l2')
#     # print pair, np.linalg.norm(X_normalized - Y_normalized)
#     print pair,np.linalg.norm(model.C[vocab.word_to_id(pair[0])]-model.C[vocab.word_to_id(pair[1])])




# i2w=dict()
#
# for key in vocab.w2i.keys():
#     i2w[vocab.w2i[key]]=key
#
# # print i2w[0]
#
#
# print vocab.word_to_id("START")
#
# train_data = load_ngram_without_corpus(sequences, vocab)
#
# train_X = train_data[:, -4:-1]
#
#
# for iter in range(10):
#
#     print "new iter", str(iter)
#
#     softmax_over_class, a, h = model.forward_minibatch(train_X[:,-3:])
#
#     print type(softmax_over_class)
#
#     labels=np.argmax(softmax_over_class,axis=1)
#     print labels
#     print [i2w[label] for label in labels]
#
#
#     # print np.asarray(labels)
#     # print np.asarray(labels).reshape(labels.shape[0],1)
#
#     print "old train X\n", train_X
#
#
#
#     train_X = np.append(train_X, np.asarray(labels).reshape(labels.shape[0], 1), 1)
#     print "new train X\n",train_X
#     print



    # labels=np.argsort(softmax_over_class)[:,-5:]
    #
    #
    # for ind,row in enumerate(train_X):
    #     print [i2w[i] for i in row]



    # labels = np.argmax(softmax_over_class, axis=0)
    # print labels.shape
    #
    # print [i2w[label] for label in labels]
    #
    # train_X = np.append(train_X, np.asarray(labels).reshape(labels.shape, 1), 1)
    #
    # labels = np.argsort(softmax_over_class)[:, -5:]
    #
    # for ind, label_seq in enumerate(labels):
    #     print sequences[ind][:-1], "\t\ttop 5 candidates (low to high prob): \t\t", [i2w[i] for i in label_seq]













# sanity check

# for seq in sequences:
#     print seq, [vocab.word_to_id(token) for token in seq]
#
# print train_data




# print "training ngram size " + str(train_data.shape)
#
# # valid_data=load_ngram(args.valid,vocab)
# # pickle.dump(valid_data, open("pickle_valid_data", "w"))
# valid_data = pickle.load(open("pickle_valid_data", "r"))
# print "valid ngram size " + str(valid_data.shape)
#
# train_X = train_data[:, :-1]
