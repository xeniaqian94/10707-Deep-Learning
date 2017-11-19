import pickle
import sys

from train import Vocab, Model
from util import *

label = sys.argv[1]  # -lr_1.0_-max_epoch_200

sequences = [['government', 'of', 'united', 'states'], ['he', 'said', 'that', 'is'], ['city', 'of', 'new', 'york'], ['of', 'new', 'york','city'],['life', 'in', 'the', 'world'],
             ['he', 'is', 'the', 'man'], ['the', 'international', 'olympic', 'committee'],['city', 'of', 'los', 'angeles'],
             ['the', 'national', 'association','of'], ['the', 'u.s.', 'and', 'japanese'], ['the', 'commerce', 'department','has'],['international','business','machines']]

model = pickle.load(open("dump/model_" + label, "r"))
vocab = pickle.load(open("dump/vocab_" + label, "r"))


i2w=dict()

for key in vocab.w2i.keys():
    i2w[vocab.w2i[key]]=key

# print i2w[0]


print vocab.word_to_id("START")

train_data = load_ngram_without_corpus(sequences, vocab)

train_X = train_data[:, -4:-1]


for iter in range(10):

    print "new iter", str(iter)

    softmax_over_class, a, h = model.forward_minibatch(train_X[:,-3:])

    print type(softmax_over_class)

    labels=np.argmax(softmax_over_class,axis=1)
    print labels
    print [i2w[label] for label in labels]


    # print np.asarray(labels)
    # print np.asarray(labels).reshape(labels.shape[0],1)

    print "old train X\n", train_X



    train_X = np.append(train_X, np.asarray(labels).reshape(labels.shape[0], 1), 1)
    print "new train X\n",train_X
    print



    # labels=np.argsort(softmax_over_class)[:,-5:]
    #
    #
    for ind,row in enumerate(train_X):
        print [i2w[i] for i in row]



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
