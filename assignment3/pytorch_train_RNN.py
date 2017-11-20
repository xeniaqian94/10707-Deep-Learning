import torch
import torch.autograd as autograd
from torch.autograd import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
from util import *

from train import Model, Vocab
import sys

torch.manual_seed(1)
vocab = pickle.load(open("dump/vocab_-lr_0.5", "r"))

# plot.py dump/log_pytorch_rnn__embed_size_16 dump/log_pytorch_rnn__embed_size_32 dump/log_pytorch_rnn__embed_size_64 dump/log_pytorch_rnn__embed_size_128 dump/log_-lr_1.0

word_to_ix = vocab.w2i

CONTEXT_SIZE = 3
EMBEDDING_DIM = int(sys.argv[1])
HIDDEN_SIZE = 128
ACTIVATION = 'linear'
VOCAB_SIZE = len(vocab.w2i.keys())
MAX_EPOCH = 100
BATCH_SIZE = 16

train_data = pickle.load(open("pickle_train_data", "r"))
print "training ngram size " + str(train_data.shape)

valid_data = pickle.load(open("pickle_valid_data", "r"))
print "valid ngram size " + str(valid_data.shape)

train_X = train_data[:, :-1]
train_Y = train_data[:, -1]

valid_X = valid_data[:, :-1]
valid_Y = valid_data[:, -1]


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, context_size, activation):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn=nn.RNN(embedding_dim, hidden_size, 1)

        self.h0=Variable(torch.randn(1,1,hidden_size))

        self.linear=nn.Linear(hidden_size,vocab_size)






        # self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, vocab_size)
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.activation = activation

    def forward(self, inputs,Y):
        embeds = self.embeddings(inputs).permute(1,0,2)  #embeds size torch.Size([3, 16, 10])

        # raw_input("embeds size "+str(embeds.size()))

        output,hn=self.rnn(embeds,self.h0)

        hn=hn.squeeze()
        # raw_input("hn size "+str(hn.size()))

        out=self.linear(hn)


        # raw_input(self.embeddings(inputs).size())
        # raw_input(embeds.size())
        # raw_input(embeds.view((-1, self.embedding_dim * self.context_size)))

        # if self.activation == 'linear':
        #     out = self.linear1(embeds.view((-1, self.embedding_dim * self.context_size)))
        # else:
        #     out = F.tanh(self.linear1(embeds.view((-1, self.embedding_dim * self.context_size))))


        # out = F.relu(self.linear1(embeds))
        # out = self.linear2(out)

        probs = out.data.numpy()

        exp_scores = np.exp(probs)

        softmax_over_class = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        n, ce_batch, sum_batch = calculate_from_softmax(softmax_over_class, Y)

        # log_probs = F.log_softmax(out)
        # probs = F.softmax(out)


        # probs = out
        #
        # probs = probs.data.numpy()
        #
        # exp_scores = np.exp(probs)
        #
        # # raw_input("exp scores shape " + str(exp_scores.shape))
        # softmax_over_class = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # sum across rows
        #
        # raw_input("softmax_over_class " + str(softmax_over_class.shape))


        return out,n,ce_batch,sum_batch


    def evaluate(self,X,Y):
        context_var = autograd.Variable(torch.LongTensor(X))
        out,n,ce_batch,sum_batch = self.forward(context_var,Y)

        ce = 1.0 / n * ce_batch
        perp = 2 ** (-1. / n * sum_batch)

        # correct_logprobs = -np.log2(probs[range(probs.shape[0]), target])
        # perp = np.average(correct_logprobs)

        return ce, perp

def calculate_from_softmax(softmax_over_class,Y):
    return softmax_over_class.shape[0],np.sum(-1.0 * np.log(softmax_over_class[np.arange(softmax_over_class.shape[0]), Y])),np.sum(np.log2(softmax_over_class[np.arange(softmax_over_class.shape[0]), Y]))


losses = []
# loss_function = nn.NLLLoss()
loss_function = nn.CrossEntropyLoss()
model = NGramLanguageModeler(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, CONTEXT_SIZE, ACTIVATION)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)




label = "_embed_size_"+str(EMBEDDING_DIM)
print "label is " + label

f_log=open("dump/log_pytorch_rnn_"+label,"w")
best_CE=float("inf")
for epoch in range(MAX_EPOCH):

    train_X, train_Y = unison_shuffled_copies(train_X, train_Y)
    f_log.write("epoch " + str(epoch + 1) + " start \n")

    CE=0
    PERP=0
    N=0

    for ind, instance_id in enumerate(range(0, train_X.shape[0], BATCH_SIZE)[:-1]):
        # print "epoch ",epoch,"ind ",ind,"total ",len(range(0, train_X.shape[0], BATCH_SIZE)[:-1])
        context=train_X[instance_id:min(instance_id + BATCH_SIZE, len(train_X))]
        target=train_Y[instance_id:min(instance_id + BATCH_SIZE, len(train_Y))]

        context_var = autograd.Variable(torch.LongTensor(context))
        target_var = autograd.Variable(torch.LongTensor(target))

        optimizer.zero_grad()
        probs,n,ce_batch,sum_batch = model(context_var,target)
        loss = loss_function(probs, target_var)
        loss.backward()
        optimizer.step()

        N +=n
        CE+=ce_batch
        PERP+=sum_batch

        exported_embeddings=model.embeddings(autograd.Variable(torch.LongTensor(np.arange(VOCAB_SIZE))))
        # raw_input(exported_embeddings.size())
        #
        # raw_input(exported_embeddings.data.numpy().shape)
        # raw_input("catch all")



    f_log.write("cross entropy for training " + str((1. / N * CE)) + "\n")
    f_log.write("perplexity for training " + str(2 ** (-1. / N * PERP)) + "\n")

    ce, perp = model.evaluate(valid_X, valid_Y)
    f_log.write("cross entropy for validation " + str(ce) + "\n")
    f_log.write("perplexity for validation " + str(perp) + "\n")
    f_log.flush()

    if best_CE>(1. / N * CE):
        best_CE=(1. / N * CE)
        pickle.dump(exported_embeddings.data.numpy(),open("dump/embedding_pytorch_rnn_"+label,"w"))



