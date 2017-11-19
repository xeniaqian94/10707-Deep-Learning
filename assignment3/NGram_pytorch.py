import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
from random import shuffle
import matplotlib.pyplot as plt

class NGramLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, context_size, act_type):
        super(NGramLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size*embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.act_type = act_type
    def forward(self, inputs):
        embeds =self.embeddings(inputs)
        if self.act_type == 'linear':
            out = self.linear1(embeds.view((-1, self.embedding_dim * self.context_size)))
        else:
            out = F.tanh(self.linear1(embeds.view((-1,self.embedding_dim * self.context_size))))
        out = self.linear2(out)
        probs = out
        # probs = F.softmax(out)
        return probs
    def get_embeds(self, inputs):
        return self.embeddings(inputs)

def evaluate(val_data, batch_size, model, loss_function):
    num_batches = val_data.shape[0]/batch_size
    model.eval()
    val_loss = torch.Tensor([0])
    for ind in range(num_batches):
        context = val_data[ind*batch_size:(ind+1)*batch_size,:-1]
        target = val_data[ind*batch_size:(ind+1)*batch_size,-1]

        context_var = autograd.Variable(torch.LongTensor(context)).cuda()
        target_var = autograd.Variable(torch.LongTensor(target)).cuda()

        probs = model(context_var)
        loss = loss_function(probs, target_var)
        val_loss += loss.cpu().data[0]
    return val_loss,num_batches
def compute_tr_perp(model):
    data_file = '../data/train.txt'
    with open('../data/vocab.pkl', 'rb') as f:
        voc_dict = pickle.load(f)
    perps = []
    with open(data_file ,'rb') as fr:
        lines = fr.readlines()
        for line in lines:
            data = extract_ngrams(line, voc_dict)
            if data.shape[0] == 0:
                continue
            sen_perp = get_perp_sentence(data, model)
            perps.append(sen_perp)

    return sum(perps)/len(perps)
def compute_val_perp(model):
    data_file = '../data/val.txt'
    with open('../data/vocab.pkl', 'rb') as f:
        voc_dict = pickle.load(f)
    perps = []
    with open(data_file ,'rb') as fr:
        lines = fr.readlines()
        for line in lines:
            data = extract_ngrams(line, voc_dict)
            if data.shape[0] == 0:
                continue
            sen_perp = get_perp_sentence(data, model)
            perps.append(sen_perp)

    return sum(perps)/len(perps)

def extract_ngrams(line, voc_dict):
    data = []
    words = line.rstrip('\r\n').lower().split(' ')
    # words = ['START', 'START', 'START'] + words + ['END']
    words = ['START'] + words + ['END']
    for start_ind in range(len(words) - 3):
        word_ind = []
        for index in range(start_ind, start_ind + 4):
            if words[index] in voc_dict:
                word_ind.append(voc_dict[words[index]])
            else:
                word_ind.append(voc_dict['UNK'])
        data.append(word_ind)
    return np.asarray(data)

def get_perp_sentence(data, model):

    context = data[:,:-1]
    target = data[:,-1]
    model.eval()
    context_var = autograd.Variable(torch.LongTensor(context)).cuda()
    probs = model(context_var)
    probs = probs.data.cpu().numpy()
    exp_scores = np.exp(probs)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log2(probs[range(probs.shape[0]),target])
    sen_perp = np.average(correct_logprobs)
    return 2**sen_perp


def word_to_idx(data_file, voc_dict):
    data = []
    with open(data_file, 'rb') as fr:
        lines = fr.readlines()
        for line in lines:
            words = line.rstrip('\r\n').lower().split(' ')
            # words = ['START', 'START', 'START'] + words + ['END']
            words = ['START'] + words + ['END']
            for start_ind in range(len(words) - 3):
                word_ind = []
                for index in range(start_ind, start_ind + 4):
                    if words[index] in voc_dict:
                        word_ind.append(voc_dict[words[index]])
                    else:
                        word_ind.append(voc_dict['UNK'])
                data.append(word_ind)
    data = np.asarray(data, dtype=np.int)
    return data

def run_mlp(hidden_size, act_type):
    tr_data = np.load('../data/tr_data.npy')
    val_data = np.load('../data/val_data.npy')

    vocab_size = 8000
    embedding_dim = 16
    # hidden_size = 128
    context_size = 3
    num_epoches = 100
    batch_size = 512

    tr_losses = []
    val_losses = []
    tr_perps = []
    val_perps = []
    model = NGramLM(vocab_size, embedding_dim, hidden_size, context_size, act_type)
    model.cuda()
    loss_function = nn.CrossEntropyLoss()
    if act_type == 'linear':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.Adam(model.parameters())


    for epoch in range(num_epoches):
        model.train()
        tr_loss = torch.Tensor([0])
        np.random.shuffle(tr_data)
        num_batches = tr_data.shape[0]/batch_size
        for ind in range(num_batches):
            context = tr_data[ind*batch_size:(ind+1)*batch_size,:-1]
            target = tr_data[ind*batch_size:(ind+1)*batch_size,-1]

            context_var = autograd.Variable(torch.LongTensor(context)).cuda()
            target_var = autograd.Variable(torch.LongTensor(target)).cuda()

            optimizer.zero_grad()
            probs = model(context_var)
            loss = loss_function(probs, target_var)
            loss.backward()
            optimizer.step()
            tr_loss += loss.cpu().data[0]

        val_loss,num_val = evaluate(val_data, batch_size, model, loss_function)
        tr_perp = compute_tr_perp(model)
        # tr_perp = 0
        val_perp = compute_val_perp(model)
        print '{}th epoch, training loss is {}, val loss is {}, perp is {}'.format(epoch, tr_loss.numpy()[0]/num_batches, val_loss.numpy()[0]/num_val, val_perp)

        tr_losses.append(tr_loss.numpy()[0]/num_batches)
        val_losses.append(val_loss.numpy()[0]/num_val)
        val_perps.append(val_perp)
        tr_perps.append(tr_perp)

    return tr_losses, val_losses, tr_perps, val_perps

def plot_32():
    hidden_sizes = [128, 256, 512]
    num_epoches = 100
    tr_losses = []
    val_losses = []
    tr_perps = []
    val_perps = []
    act_type = 'linear'
    for hidden_size in hidden_sizes:
        tr_loss, val_loss, tr_perp, val_perp = run_mlp(hidden_size, act_type)
        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        tr_perps.append(tr_perp)
        val_perps.append(val_perp)
    iters = range(num_epoches)
    plt.figure()
    for i,hidden_size in enumerate(hidden_sizes):
        plt.plot(iters, tr_losses[i], label='HiddenSize: {} trn loss'.format(hidden_size))
        plt.plot(iters, val_losses[i], label='HiddenSize: {} val loss'.format(hidden_size))
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.legend()
    plt.savefig('../figs/32-loss.png', dpi=300)

    plt.figure()
    for i,hidden_size in enumerate(hidden_sizes):
        plt.plot(iters, tr_perps[i], label='HiddenSize: {} tr perp'.format(hidden_size))
        plt.plot(iters, val_perps[i], label='HiddenSize: {} val perp'.format(hidden_size))
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.savefig('../figs/32-perp.png', dpi=300)

    
def plot_33():
    hidden_sizes = [128, 256, 512]
    num_epoches = 100
    tr_losses = []
    val_losses = []
    tr_perps = []
    val_perps = []
    act_type = 'non-linear'
    for hidden_size in hidden_sizes:
        tr_loss, val_loss, tr_perp, val_perp = run_mlp(hidden_size, act_type)
        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        tr_perps.append(tr_perp)
        val_perps.append(val_perp)
    iters = range(num_epoches)
    plt.figure()
    for i,hidden_size in enumerate(hidden_sizes):
        plt.plot(iters, tr_losses[i], label='HiddenSize: {} trn loss'.format(hidden_size))
        plt.plot(iters, val_losses[i], label='HiddenSize: {} val loss'.format(hidden_size))
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.legend()
    plt.savefig('../figs/33-loss.png', dpi=300)

    plt.figure()
    for i,hidden_size in enumerate(hidden_sizes):
        plt.plot(iters, tr_perps[i], label='HiddenSize: {} tr perp'.format(hidden_size))
        plt.plot(iters, val_perps[i], label='HiddenSize: {} val perp'.format(hidden_size))
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.savefig('../figs/33-perp.png', dpi=300)
def plot_33_comp():
    hidden_size = 128
    num_epoches = 100
    tr_losses = []
    val_losses = []
    tr_perps = []
    val_perps = []
    act_types = ['linear','non-linear']
    for act_type in act_types:
        tr_loss, val_loss, tr_perp, val_perp = run_mlp(hidden_size, act_type)
        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        tr_perps.append(tr_perp)
        val_perps.append(val_perp)
    iters = range(num_epoches)
    plt.figure()
    for i,act_type in enumerate(act_types):
        plt.plot(iters, tr_losses[i], label='{} act trn loss'.format(act_type))
        plt.plot(iters, val_losses[i], label='{} act val loss'.format(act_type))
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.legend()
    plt.savefig('../figs/33-loss-comp.png', dpi=300)

    plt.figure()
    for i,act_type in enumerate(act_types):
        plt.plot(iters, tr_perps[i], label='{} tr perp'.format(act_type))
        plt.plot(iters, val_perps[i], label='{} val perp'.format(act_type))
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.savefig('../figs/33-perp-comp.png', dpi=300)


if __name__ == '__main__':
    plot_32()
    # plot_33()
    # plot_33_comp()


    # hidden_sizes = [128, 256, 512]
    # num_epoches = 30
    # tr_losses = []
    # val_losses = []
    # tr_perps = []
    # val_perps = []
    # for hidden_size in hidden_sizes:
    #     tr_loss, val_loss, val_perp = run_mlp(hidden_size)
    #     tr_losses.append(tr_loss)
    #     val_losses.append(val_loss)
    #     val_perps.append(val_perp)
    # iters = range(num_epoches)
    # plt.figure()
    # for i,hidden_size in enumerate(hidden_sizes):
    #     plt.plot(iters, tr_losses[i], label='HiddenSize: {} trn loss'.format(hidden_size))
    #     plt.plot(iters, val_losses[i], label='HiddenSize: {} val loss'.format(hidden_size))
    # plt.xlabel('Epoch')
    # plt.ylabel('Cross-entropy loss')
    # plt.legend()
    # plt.savefig('../figs/32-loss.png', dpi=300)
    #
    # plt.figure()
    # for i,hidden_size in enumerate(hidden_sizes):
    #     plt.plot(iters, tr_perps[i], label='HiddenSize: {} tr perp'.format(hidden_size))
    #     plt.plot(iters, val_perps[i], label='HiddenSize: {} val perp'.format(hidden_size))
    # plt.xlabel('Epoch')
    # plt.ylabel('Perplexity')
    # plt.legend()
    # plt.savefig('../figs/32-perp.png', dpi=300)



    # model = run_mlp(hidden_size)
    #
    # model.eval()
    # voc_list = []
    # with open('../data/vocab.pkl', 'rb') as f:
    #     voc_dict = pickle.load(f)
    #
    # for voc in voc_dict:
    #     voc_list.append(voc)
    # shuffle(voc_list)
    # rand_words = voc_list[:500]
    # rand_inds = []
    # for word in rand_words:
    #     rand_inds.append(voc_dict[word])
    # rand_inds = np.asarray(rand_inds)
    #
    # context_var = autograd.Variable(torch.LongTensor(rand_inds)).cuda()
    # word_embeddings = model.get_embeds(context_var)
    # # print rand_inds.shape
    # # print word_embeddings.cpu().data.shape
    #
    # np.save('../data/t-sne.npy',word_embeddings.cpu().data.numpy())






    # for model_type in model_types:
    #     tr_loss, val_loss, val_perp = run_mlp(hidden_size, model_type)
    #     tr_losses.append(tr_loss)
    #     val_losses.append(val_loss)
    #     val_perps.append(val_perp)
    # iters = range(num_epoches)
    # plt.figure()
    # for i,model_type in enumerate(model_types):
    #     plt.plot(iters, tr_losses[i], label='{} act trn loss'.format(model_type))
    #     plt.plot(iters, val_losses[i], label='{} act val loss'.format(model_type))
    # plt.xlabel('Epoch')
    # plt.ylabel('Cross-entropy loss')
    # plt.legend()
    # plt.savefig('../figs/33-loss-comp.png', dpi=300)
    #
    # plt.figure()
    # for i,model_type in enumerate(model_types):
    #     plt.plot(iters, val_perps[i], label='{} act val perp'.format(model_type))
    # plt.xlabel('Epoch')
    # plt.ylabel('Perplexity')
    # plt.legend()
    # plt.savefig('../figs/33-perp-comp.png', dpi=300)


    # for hidden_size in hidden_sizes:
    #     tr_loss, val_loss, val_perp = run_mlp(hidden_size)
    #     tr_losses.append(tr_loss)
    #     val_losses.append(val_loss)
    #     val_perps.append(val_perp)
    # iters = range(num_epoches)
    # plt.figure()
    # for i,hidden_size in enumerate(hidden_sizes):
    #     plt.plot(iters, tr_losses[i], label='HiddenSize: {} trn loss'.format(hidden_size))
    #     plt.plot(iters, val_losses[i], label='HiddenSize: {} val loss'.format(hidden_size))
    # plt.xlabel('Epoch')
    # plt.ylabel('Cross-entropy loss')
    # plt.legend()
    # plt.savefig('../figs/33-loss.png', dpi=300)
    #
    # plt.figure()
    # for i,hidden_size in enumerate(hidden_sizes):
    #     plt.plot(iters, val_perps[i], label='HiddenSize: {} val perp'.format(hidden_size))
    # plt.xlabel('Epoch')
    # plt.ylabel('Perplexity')
    # plt.legend()
    # plt.savefig('../figs/33-perp.png', dpi=300)
