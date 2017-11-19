import sys
import argparse



from itertools import count, izip, chain
import time
from util import *
import pickle



class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.iteritems()}

    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.iteritems()}

    @classmethod
    def from_corpus(cls, corpus, top=8000):
        data = read_corpus(corpus)
        freqs = Counter(chain(*data))
        sorted_freqs = sorted(freqs.iteritems(), key=operator.itemgetter(1), reverse=True)

        w2i = defaultdict(count(0).next)
        w2i["UNK"]

        print "top 10 tokens after UNK"

        print sorted_freqs[:10]

        [w2i[key] for (key, value) in sorted_freqs[:top - 1] if value >= 1]  # eliminate singleton

        vocab = Vocab(w2i)
        return vocab, data

    def size(self):
        return len(self.w2i.keys())

    def word_to_id(self, word):
        if word in self.w2i:
            return self.w2i[word]
        else:
            return 0  # self.w2i["UNK"]


def tanh(x):
    return np.divide(np.exp(np.add(x, x)) - 1, np.exp(np.add(x, x)) + 1)


def deact_tanh(g_a, a):
    return 1 - np.multiply(g_a, g_a)


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', type=str, help='training path', default="train.txt")
    parser.add_argument('-valid', type=str, help='validation path', default="val.txt")
    parser.add_argument('-vocab', type=str, help='path of the serialized vocabulary')
    parser.add_argument('-vocab_size', type=int, default=8000, help='vocab size')

    # parser.add_argument('-batch_size', default=32, type=int, help='batch size')
    parser.add_argument('-embed_size', default=16, type=int, help='size of word embeddings')
    parser.add_argument('-hidden_size', default=128, type=int, help='hidden layer size')
    parser.add_argument('-num_layer', type=int, help="if 1 single layer, 2 then 2-layer network", default=1)
    parser.add_argument('-ngram', default=4, type=int)
    parser.add_argument('-lr', type=float, help="learning rate", default=1.0)
    parser.add_argument('-lbd', type=float, help="regularization term", default=0.001)
    # parser.add_argument('-momentum', type=float, help="average gradient", default=0.5)
    parser.add_argument('-minibatch_size', type=int, help="minibatch_size", default=1024)
    parser.add_argument('-batch_normalization', type=bool, help="whether do batch normalization or not ", default=True)
    parser.add_argument('-activation', type=str, help="which activation to use ", default="none")
    parser.add_argument('-max_epoch', type=int, default=100)

    args = parser.parse_args()
    np.random.seed(int(time.time()))

    return args


class Model:
    def __init__(self, args):  # here len(dimension_list) should equal to num_layer+2
        # self.L_num_layer = args.num_layer  # num_layer=L


        self.W = [[]]
        self.b = [[]]
        self.lr = args.lr
        self.lbd = args.lbd
        self.L_num_layer = args.num_layer

        # self.W_epsilon = [[]]
        # self.prev_W = None
        # self.prev_b = None
        # self.momentum = momentum
        # self.gamma = [[]]
        # self.prev_gamma = [[]]
        # self.beta = [[]]
        # self.prev_beta = [[]]
        # self.global_mean = [[]]
        # self.global_variance = [[]]
        # self.new_W = []
        # self.new_b = []
        # self.sum_gradient_W = []
        # self.sum_gradient_b = []
        # self.old_W = []
        # self.old_b = []

        self.batch_normalization = args.vocab_size

        self.C = np.random.normal(0, 1e-5, (args.vocab_size, args.embed_size)).astype(np.float32)

        self.num_class = args.vocab_size

        self.dimension_list = [args.embed_size, args.hidden_size, self.num_class]

        self.ngram = args.ngram
        self.minibatch_size = args.minibatch_size

        for i in range(1, self.L_num_layer + 2):

            if i == 1:
                ngram_array = []
                for j in range(args.ngram - 1):
                    ngram_array += [
                        np.random.normal(0, 1e-3, (self.dimension_list[i], self.dimension_list[i - 1])).astype(
                            np.float32)]
                self.W += [ngram_array]
            else:
                self.W += [
                    np.random.normal(0, 0.17, (self.dimension_list[i], self.dimension_list[i - 1])).astype(np.float32)]

            self.b += [np.random.normal(0, 1e-10, self.dimension_list[i]).astype(np.float32)]
        self.print_parameters()

    def print_parameters(self):
        print "self.W[1][1].shape " + str(self.W[1][1].shape)
        print "self.W[2].shape " + str(self.W[2].shape)
        print "self.b[1].shape " + str(self.b[1].shape)
        print "self.C.shape " + str(self.C.shape)



    #
    #
    #     def forward(self, x, activation=None):
    #         h = [x]
    #         a = [[]]
    #
    #         for i in range(1, self.L_num_layer + 2):
    #
    #             if (i==1):
    #                 a+=[np.dot()]
    #             a += [np.dot(self.W[i], h[i - 1]) + self.b[i]]
    #             # print "current a_" + str(i) + " shape " + str(a[-1].shape)
    #             if i == self.L_num_layer + 1:
    #                 break
    #             h += [activation(a[i])]  # h_k=g(a_k)
    #             # print "current h_" + str(i) + " shape " + str(h[-1].shape)
    #         # print "final a_" + str(i) + " shape " + str(a[-1].shape)
    #         softmax_over_class = softmax(a[-1])
    #         # print "softmax value " + str(softmax_over_class) + " " + str(np.sum(softmax_over_class))
    #         return softmax_over_class, a, h
    #
    #     def forward_gradient_check(self, x, W_epsilon, activation=sigmoid):
    #         h = [x]
    #         a = [[]]
    #
    #         for i in range(1, self.L_num_layer + 2):
    #             a += [np.dot(W_epsilon[i], h[i - 1]) + self.b[i]]
    #             # print "current a_" + str(i) + " shape " + str(a[-1].shape)
    #             if i == self.L_num_layer + 1:
    #                 break
    #             h += [sigmoid(a[i])]  # h_k=g(a_k)
    #             # print "current h_" + str(i) + " shape " + str(h[-1].shape)
    #         # print "final a_" + str(i) + " shape " + str(a[-1].shape)
    #         softmax_over_class = softmax(a[-1])
    #         # print "softmax value " + str(softmax_over_class) + " " + str(np.sum(softmax_over_class))
    #         return softmax_over_class, a, h
    #
    #     def gradient_check(self, i, delta_Wi, x, y):
    #         new_W_plus_epsilon = self.W
    #         new_W_plus_epsilon[i] += self.W_epsilon[i]
    #         f_W_plus_epsilon, a, h = self.forward_gradient_check(x, new_W_plus_epsilon)
    #
    #         new_W_minus_epsilon = self.W
    #         new_W_minus_epsilon[i] -= self.W_epsilon[i]
    #         f_W_minus_epsilon, a, h = self.forward_gradient_check(x, new_W_minus_epsilon)
    #
    #         print str(delta_Wi.shape) + " " + str(self.W_epsilon[i].shape) + " sanity check shape " + str(
    #             (f_W_plus_epsilon - f_W_minus_epsilon).shape)
    #
    #     def back_prop(self, x, y, f_x, a, h, deactivation=derivative_sigmoid):
    #         e_y = np.zeros(self.num_class)
    #         e_y[y] = 1  # one hot class label
    #
    #         gradient_a = [None] * (self.L_num_layer + 2)
    #         gradient_W = [None] * (self.L_num_layer + 2)
    #         gradient_b = [None] * (self.L_num_layer + 2)
    #         gradient_h = [None] * (self.L_num_layer + 1)
    #
    #         gradient_a[self.L_num_layer + 1] = -1 * (e_y - f_x)
    #         # raw_input("gradient_a \n"+str(gradient_a[self.L_num_layer + 1]))
    #
    #         for k in range(self.L_num_layer + 1, 0, -1):
    #             gradient_W[k] = np.outer(gradient_a[k], h[k - 1])
    #             # raw_input(str(k)+" gradient_W[k] h[k-1] \n" + str(gradient_W[k][0])+"\n"+str(h[k-1]))
    #             gradient_b[k] = gradient_a[k]
    #
    #             if k == 1:
    #                 break
    #
    #             gradient_h[k - 1] = np.dot(self.W[k].T, gradient_a[k])
    #             # raw_input(str(k)+" gradient_h[k - 1] \n" + str(gradient_h[k - 1]))
    #
    #             gradient_a[k - 1] = np.multiply(gradient_h[k - 1], deactivation(h[k - 1], a[k - 1]))
    #             # raw_input(str(k)+" gradient_a[k - 1] \n"+str(gradient_a[k - 1]))
    #
    #         # eliminate the regularization term here
    #         # print gradient_b
    #
    #         for i in range(1, self.L_num_layer + 2):
    #             delta_W_i = -1.0 * gradient_W[i] - 2.0 * self.lbd * self.W[i]
    #             delta_b_i = -1 * gradient_b[i] - 2.0 * self.lbd * self.b[i]
    #
    #             if not self.prev_W == None and self.momentum > 0:
    #                 delta_W_i -= 1.0 * self.momentum * self.prev_W[i]
    #                 delta_b_i -= 1.0 * self.momentum * self.prev_b[i]
    #
    #             self.W[i] = self.W[i] + self.lr * delta_W_i
    #             self.b[i] = self.b[i] + self.lr * delta_b_i
    #
    #         self.prev_W = gradient_W
    #         self.prev_b = gradient_b
    #
    #     def update(self, x, y, activation="sigmoid"):  # y is the final labely
    #         if activation == "sigmoid":
    #             f_x, a, h = self.forward(x)
    #             # print f_x.shape
    #             self.back_prop(x, y, f_x, a, h)
    #         elif activation == "tanh":
    #             f_x, a, h = self.forward(x, tanh)
    #             self.back_prop(x, y, f_x, a, h, deact_tanh)
    #         elif activation == "ReLU":
    #             f_x, a, h = self.forward(x, ReLU)
    #             self.back_prop(x, y, f_x, a, h, deact_ReLU)
    #
    def forward_minibatch(self, X, activation=None):

        h = [[]]
        for i in range(self.ngram - 1):
            h[0] += [self.C[X[:, i]].T]

        a = [[]]

        for i in range(1, self.L_num_layer + 2):

            if (i == 1):

                Wh = np.dot(self.W[i][0], h[i - 1][0])
                for j in range(1, self.ngram - 1):
                    Wh += np.dot(self.W[i][j], h[i - 1][j])
                a += [(Wh + np.tile(self.b[i].reshape(self.b[i].shape[0], 1),
                                    (1, h[i - 1][0].shape[
                                        -1]))).T]  # a[-1] dimension = # out * # instance
            else:
                a += [(np.dot(self.W[i], h[i - 1]) + np.tile(self.b[i].reshape(self.b[i].shape[0], 1), (
                    1, h[i - 1].shape[-1]))).T]  # resulting a is #hidden unit * minibatch size

            if i == self.L_num_layer + 1:
                break
            if activation:
                h += [activation(a[-1]).T]
            else:
                h += [a[-1].T]

        softmax_over_class = softmax(a[-1].T).T  # softmax_over_class = #intances * # dimension

        return softmax_over_class, a, h

    def store_old_parameter(self):
        return

    #
    def back_prop_minibatch(self, X, Y, softmax_over_class, a, h, deactivation=None):
        num_instance = X.shape[0]

        f_X = softmax_over_class

        e_y = np.zeros([num_instance, self.num_class])
        # raw_input(type(list(Y)))

        # print list(range(len(Y))), list(Y)
        e_y[list(range(len(Y))), list(Y)] = 1  # one hot class label

        gradient_a = [None] * (
            self.L_num_layer + 2)  # except each is now not arraysize (num_class,)  but (num_instance,num_class)
        gradient_W = [None] * (self.L_num_layer + 2)
        gradient_W[1] = [None] * (self.ngram - 1)
        gradient_b = [None] * (self.L_num_layer + 2)
        gradient_h = [None] * (self.L_num_layer + 1)
        gradient_C = [None] * (self.ngram - 1)

        gradient_a[self.L_num_layer + 1] = -1 * (e_y - f_X)

        for k in range(self.L_num_layer + 1, 0, -1):

            # print "gradient_a[k].shape " + str(gradient_a[k].shape)
            # if not k == 1:
            #     print "h[k - 1].shape " + str(h[k - 1].shape)


            # raw_input()

            if k == 2:
                # gradient_W[k] = np.einsum('ki,jk->kij', gradient_a[k], h[k - 1],optimize=True)
                # gradient_W[k] = contract('ki,jk->kij', gradient_a[k], h[k - 1])
                gradient_W[k] = np.dot(h[k - 1], gradient_a[k]).T

            else:
                for i in range(self.ngram - 1):
                    # gradient_W[k][i] = np.einsum('ki,jk->kij', gradient_a[k], h[k - 1][i],optimize=True) # g_a num_instance * n_output

                    # gradient_W[k][i] = contract('ki,jk->kij', gradient_a[k],h[k - 1][i])

                    gradient_W[k][i] = np.dot(h[k - 1][i], gradient_a[k]).T

                    # print "str(gradient_W[k][i].shape) " + str(k) + " " + str(i) + " " + str(gradient_W[k][i].shape)
                    gradient_C[i] = np.dot(self.W[k][i].T, gradient_a[k].T)  # emb_size * num_instances

            # if not k == 1:
            #     print "gradient_W[k].shape " + str(gradient_W[k].shape)

            gradient_b[k] = gradient_a[k]

            if k == 1:
                break

            gradient_h[k - 1] = np.dot(self.W[k].T, gradient_a[k].T)  # num_input * num_instances
            # print "gradient_h[k - 1].shape " + str(gradient_h[k - 1].shape)

            if deactivation:
                gradient_a[k - 1] = np.multiply(gradient_h[k - 1], deactivation(h[k - 1], a[k - 1])).T
            else:
                gradient_a[k - 1] = gradient_h[k - 1].T

                # print "gradient_a[k - 1].shape " + str(gradient_a[k - 1].shape)

        for i in range(1, self.L_num_layer + 2):
            if i == 1:
                for j in range(self.ngram - 1):
                    # print gradient_W[1][2].shape
                    # delta_W_i = -1.0 * np.mean(gradient_W[i][j], axis=0) - 2.0 * self.lbd * self.W[i][j]
                    delta_W_i = -1.0 * gradient_W[i][j] / self.minibatch_size - 2.0 * self.lbd * self.W[i][j]

                    self.W[i][j] = self.W[i][j] + self.lr * delta_W_i
                    self.C[X[:, j]] -= self.lr * gradient_C[j].T / gradient_C[j].shape[1]
            else:
                # delta_W_i = -1.0 * np.mean(gradient_W[i], axis=0) - 2.0 * self.lbd * self.W[i]
                delta_W_i = -1.0 * gradient_W[i] / self.minibatch_size - 2.0 * self.lbd * self.W[i]

                self.W[i] = self.W[i] + self.lr * delta_W_i

            # if len(self.prev_W) == len(self.W) and self.momentum > 0:
            #     delta_W_i -= 1.0 * self.momentum * self.prev_W[i]
            #     delta_b_i -= 1.0 * self.momentum * self.prev_b[i]


            delta_b_i = -1 * np.mean(gradient_b[i], axis=0) - 2.0 * self.lbd * self.b[i]
            self.b[i] = self.b[i] + self.lr * delta_b_i



            # self.prev_W += [np.mean(gradient_W[i], axis=0)]
            # self.prev_b += [np.mean(gradient_b[i], axis=1)]

    def average_cross_entropy_batch(self, softmax_over_class, Y):  # softmax_over_class = # dimension * # instance

        ce_batch = np.sum(-1.0 * np.log(softmax_over_class[np.arange(softmax_over_class.shape[0]), Y]))

        return softmax_over_class.shape[0], ce_batch

    def log2P_sum_batch(self, softmax_over_class, Y):
        sum_batch = np.sum(np.log2(softmax_over_class[np.arange(softmax_over_class.shape[0]), Y]))
        return softmax_over_class.shape[0], sum_batch

    def validate(self, X, Y):
        softmax_over_class, a, h = self.forward_minibatch(X)

        n, ce_batch = self.average_cross_entropy_batch(softmax_over_class, Y)
        # print "\n\ncurrent batch avg_ce_batch " + str(avg_ce_batch) + "\n\n"
        n, sum_batch = self.log2P_sum_batch(softmax_over_class, Y)

        ce = 1.0 / n * ce_batch
        perp = 2 ** (-1. / n * sum_batch)

        return ce, perp

    def update_minibatch(self, X, Y, activation_str="none"):  # y is the final labely

        if activation_str == "none":
            softmax_over_class, a, h = self.forward_minibatch(X)

            n, ce_batch = self.average_cross_entropy_batch(softmax_over_class, Y)
            # print "\n\ncurrent batch avg_ce_batch " + str(avg_ce_batch) + "\n\n"
            n, sum_batch = self.log2P_sum_batch(softmax_over_class, Y)

            # print "a and h shape"
            # print a[-1].shape
            # print h[-1].shape

            # self.back_prop_minibatch(X, Y, f_X, a, h, out, variable_cache, derivative_sigmoid)



            # raw_input("check here ...")
            self.back_prop_minibatch(X, Y, softmax_over_class, a, h)

        elif activation_str == "tanh":
            softmax_over_class, a, h = self.forward_minibatch(X, tanh)
            n, ce_batch = self.average_cross_entropy_batch(softmax_over_class, Y)
            # print "\n\ncurrent batch avg_ce_batch " + str(avg_ce_batch) + "\n\n"
            n, sum_batch = self.log2P_sum_batch(softmax_over_class, Y)
            self.back_prop_minibatch(X, Y, softmax_over_class, a, h, deact_tanh)

        return n, sum_batch, ce_batch


        # elif activation_str == "sigmoid":
        #     f_X, a, h, out, variable_cache = self.forward_minibatch(X, sigmoid)
        #     self.back_prop_minibatch(X, Y, f_X, a, h, out, variable_cache, derivative_sigmoid)
        # elif activation_str == "tanh":
        #     f_X, a, h, out, variable_cache = self.forward_minibatch(X, tanh)
        #     self.back_prop_minibatch(X, Y, f_X, a, h, out, variable_cache, deact_tanh)
        # elif activation_str == "ReLU":
        #     f_X, a, h, out, variable_cache = self.forward_minibatch(X, ReLU)
        #     self.back_prop_minibatch(X, Y, f_X, a, h, out, variable_cache, deact_ReLU)


#
#     def reset_sum_gradient(self):
#         self.sum_gradient_W = [None] * len(self.W)
#         self.sum_gradient_b = [None] * len(self.b)
#
#     def store_old_parameter(self):
#         self.old_W = self.W
#         self.old_b = self.b
#
#     def eval_valid(self, valid_X, valid_Y, activation_str=sigmoid, minibatch=1):
#
#         if activation_str == "sigmoid":
#             activation = sigmoid
#         elif activation_str == "ReLU":
#             activation = ReLU
#         elif activation_str == "tanh":
#             activation = tanh
#
#         h = [valid_X.T]
#         a = [[]]
#         out = [[]]
#
#         for i in range(1, self.L_num_layer + 2):
#             # print np.dot(self.W[i], h[i - 1]).shape
#             # print np.tile(self.b[i].reshape(self.b[i].shape[0], 1), (1, h[i - 1].shape[1])).shape
#             a += [
#                 np.dot(self.W[i], h[i - 1]) + np.tile(self.b[i].reshape(self.b[i].shape[0], 1), (1, h[i - 1].shape[1]))]
#             # print "current a_" + str(i) + " shape " + str(a[-1].shape)
#
#             if minibatch > 1 and self.batch_normalization:
#
#                 X = a[-1].T
#
#                 mu = np.mean(X, axis=0)  # (dimension,)
#
#                 var = np.var(X, axis=0)
#
#                 X_norm = (X - self.global_mean[i]) / np.sqrt(self.global_variance[i] + 1e-8)
#
#                 out += [
#                     np.tile(self.gamma[i], (X_norm.shape[0], 1)) * X_norm + np.tile(self.beta[i], (X_norm.shape[0], 1))]
#
#                 if i == self.L_num_layer + 1:
#                     break
#
#                 h += [activation(out[-1].T)]  # h_k=g(a_k)
#
#         if minibatch > 1 and self.batch_normalization:
#             softmax_over_class = softmax(out[-1].T).T
#         else:
#             softmax_over_class = softmax(a[-1]).T
#
#         one_hot_y = self.from_Y_onehot(valid_Y)
#
#         # max_prob_class=np.sum(softmax_over_class, axis=1)
#         classification_error = np.sum(np.argmax(softmax_over_class, axis=1) != (np.asarray(valid_Y))) * 1.0 / len(
#             valid_Y)
#
#         return -1.0 * np.sum(np.multiply(np.log(softmax_over_class), one_hot_y)) / softmax_over_class.shape[
#             0], classification_error
#
#     def from_Y_onehot(self, valid_Y):
#         row = range(len(valid_Y))
#         col = np.asarray(valid_Y, dtype=int)
#         one_hot_y = np.zeros([len(valid_Y), self.num_class])
#         one_hot_y[row, col] = 1
#         return one_hot_y
#
#         # def store_global_mean_variance(self, trainX):

def save_model(model, vocab, label):
    pickle.dump(model, open("dump/model_" + label,"w"))
    pickle.dump(vocab, open("dump/vocab_" + label,"w"))


if __name__ == '__main__':

    args = init_config()

    vocab, data = Vocab.from_corpus(args.train, args.vocab_size)
    pickle.dump(vocab, open("pickle_vocab", "w"))
    # vocab = pickle.load(open("pickle_vocab", "r"))
    print ("vocab size " + str(vocab.size()))

    # train_data=load_ngram(args.train,vocab)
    # pickle.dump(train_data, open("pickle_train_data", "w"))
    train_data = pickle.load(open("pickle_train_data", "r"))
    print "training ngram size " + str(train_data.shape)

    # valid_data=load_ngram(args.valid,vocab)
    # pickle.dump(valid_data, open("pickle_valid_data", "w"))
    valid_data = pickle.load(open("pickle_valid_data", "r"))
    print "valid ngram size " + str(valid_data.shape)

    train_X = train_data[:, :-1]
    train_Y = train_data[:, -1]

    valid_X = valid_data[:, :-1]
    valid_Y = valid_data[:, -1]

    plot_perplexity_valid = []
    speed = []
    model = Model(args)
    start = time.time()
    label = "_".join(sys.argv[1:])
    print "label is " + label

    f_log=open("dump/log_"+label,"w")

    best_ce = float("inf")
    best_perp = float("inf")

    for epoch in range(args.max_epoch):

        train_X, train_Y = unison_shuffled_copies(train_X, train_Y)
        then = time.time()
        f_log.write("epoch " + str(epoch + 1) + " start have spent " + str(then - start) + " " + str(
            (then - start) * 1.0 / 1000)+"\n")

        # count = 0

        N = 0
        Perplexity_sum = 0.0
        Ce_sum = 0.0

        for ind, instance_id in enumerate(range(0, train_X.shape[0], args.minibatch_size)[:-1]):
            # print str(ind * 1.0 / len(
            #     range(0, train_X.shape[0], args.minibatch_size)[:-1])) + " has finished for this epoch "
            n, perplexity_sum, ce_sum = model.update_minibatch(
                train_X[instance_id:min(instance_id + args.minibatch_size, len(train_X))],
                train_Y[instance_id:min(instance_id + args.minibatch_size, len(train_Y))],
                args.activation)
            N += n
            Perplexity_sum += perplexity_sum
            Ce_sum += ce_sum

        # this_ce = (1. / N * Ce_sum)
        # this_perp = 2 ** (-1. / N * Perplexity_sum)

        f_log.write("cross entropy for training " + str((1. / N * Ce_sum))+"\n")
        f_log.write("perplexity for training " + str(2 ** (-1. / N * Perplexity_sum))+"\n")

        ce, perp = model.validate(valid_X, valid_Y)
        f_log.write("cross entropy for validation " + str(ce)+"\n")
        f_log.write("perplexity for validation " + str(perp)+"\n")
        f_log.flush()



        if (ce<best_ce) and (perp<best_perp):
            best_ce=ce
            best_perp=perp
            save_model(model, vocab, label)





        #     count += 1
        #     if (count > 10):
        #         break
        # break


        # print "evaluating valid current learning rate " + str(model.lr)
        # then = time.time()
        # cross_entropy, classification_error = model.eval_valid(valid_X, valid_Y, args.activation,
        #                                                        args.minibatch_size)
        # # if old_classification_error < cross_entropy or old_classification_error < classification_error:
        # plot_epoch_ce_valid += [cross_entropy]
        # plot_epoch_cr_valid += [classification_error]
        # print "cross_entropy classification_error valid " + str(cross_entropy) + " " + str(classification_error)
        #
        # cross_entropy, classification_error = model.eval_valid(train_X, train_Y, args.activation,
        #                                                        args.minibatch_size)
        # plot_epoch_ce_train += [cross_entropy]
        # plot_epoch_cr_train += [classification_error]
        # print "cross_entropy classification_error train " + str(cross_entropy) + " " + str(classification_error)
        #
        # cross_entropy, classification_error = model.eval_valid(test_X, test_Y, args.activation, args.minibatch_size)
        # plot_epoch_ce_test += [cross_entropy]
        # plot_epoch_cr_test += [classification_error]
        # print "cross_entropy classification_error test " + str(cross_entropy) + " " + str(classification_error)
        #
        # print "evaluating valid end in " + str(time.time() - then)
















    # untruncated vocab
    # plot_ngram(data)

    # truncated vocab
    # plot_ngram(data, vocab)
