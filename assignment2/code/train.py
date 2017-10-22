import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time
import math
import argparse
import matplotlib.pyplot as plt
import pickle


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    try:
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    except:
        print "exception softmax"
        print x
        sys.exit(0)


def sigmoid(x):
    try:
        g_a = 1 / (1 + np.exp(-x))
        # print "forward "+str(g_a[:5])
        return g_a
    except:
        print "exception sigmoid!"
        sys.exit(0)


def derivative_sigmoid(g_a, a):
    return np.multiply(g_a, 1 - g_a)


def tanh(x):
    # print "using tanh " + str(np.divide(np.exp(np.add(x, x)) - 1, np.exp(np.add(x, x)) + 1)[:5])
    # print np.add(x,x)[:5]
    # print (np.exp(np.add(x,x)) - 1 / np.exp(np.add(x,x)) + 1)[:5]
    return np.divide(np.exp(np.add(x, x)) - 1, np.exp(np.add(x, x)) + 1)


def deact_tanh(g_a, a):
    # print "using deact tanh " + str(g_a[:5])
    return 1 - np.multiply(g_a, g_a)


def tabulate(x, y, f):
    return np.vectorize(f)(x, y)


def ReLU(x):
    # print "using ReLU"
    return tabulate(x, np.zeros(x.shape), max)


def positive(x, y):
    if x > 0:
        return 1
    else:
        return 0


def deact_ReLU(g_a, a):
    a = np.asarray(a)
    # print a
    # print "using deact ReLU"
    return tabulate(a, np.zeros(a.shape), positive)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class Model:
    def __init__(self, n_visible, n_hidden, k, lr, minibatch_size):
        self.k = k
        self.W = None
        self.hbias = None
        self.vbias = None
        self.lr = lr

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.W = np.random.normal(0, 0.08, (n_hidden, n_visible))
        self.hbias = np.zeros(n_hidden)
        self.vbias = np.zeros(n_visible)

    def pre_activation_h_given_v(self, v):
        if len(v.shape) == 1:
            return np.dot(self.W, v) + self.hbias
        return np.dot(self.W, v.T).T + np.tile(self.hbias, [v.shape[0], 1])

    def prob_h_given_v(self, v):
        return sigmoid(self.pre_activation_h_given_v(v))

    def sample_h_given_v(self, v):
        prob_h = self.prob_h_given_v(v)
        return np.random.binomial(1, prob_h, prob_h.shape[0]), prob_h

    def pre_activation_v_given_h(self, h):
        if len(h.shape) == 1:
            return np.dot(self.W.T, h) + self.vbias
        return np.dot(h, self.W) + np.tile(self.vbias, [h.shape[0], 1])

    def prob_v_given_h(self, h):
        return sigmoid(self.pre_activation_v_given_h(h))

    def sample_v_given_h(self, h):
        prob_v = self.prob_v_given_h(h)
        return np.random.binomial(1, prob_v, prob_v.shape[0]), prob_v

    def gibbs_negative_sampling(self, v, k=0):
        v_sample = v  # v0
        h_sample,prob_h = self.sample_h_given_v(v_sample)  # h0
        h0 = h_sample
        v0 = v_sample
        if k == 0:
            this_k = self.k
        else:
            this_k = k

        for i in range(1, this_k + 1):
            v_sample, prob_v = self.sample_v_given_h(h_sample)
            h_sample, prob_h = self.sample_h_given_v(v_sample)
        return [h0, v0, h_sample, v_sample, prob_h, prob_v]

    def reconstruction(self, v):
        # raw_input(v.shape)
        prob_h = self.prob_h_given_v(v)
        # raw_input(prob_h.shape)

        prob_v = self.prob_v_given_h(prob_h)
        # raw_input(prob_v.shape)
        return prob_v

    def update(self, x):
        h0, v0, h_tilda, v_tilda, h_prob, v_prob = self.gibbs_negative_sampling(x)

        # save old parameters if necessary
        self.W += self.lr * (np.outer(h0, v0.T) - np.outer(h_tilda, v_tilda.T))
        self.hbias += self.lr * (h0 - h_tilda)
        self.vbias += self.lr * (v0 - v_tilda)

    def update_minibatch(self, x):
        h0, v0, h_tilda, v_tilda, h_prob, v_prob = self.gibbs_negative_sampling(x)

        # save old parameters if necessary
        self.W += self.lr * (np.outer(h0, v0.T) - np.outer(h_tilda, v_tilda.T))
        self.hbias += self.lr * (h0 - h_tilda)
        self.vbias += self.lr * (v0 - v_tilda)

    def eval(self, X):

        X_tilda = self.reconstruction(X)
        cross_entropy = -1.0 / X_tilda.shape[0] * np.sum(X * np.log(X_tilda) + (1 - X) * np.log(1 - X_tilda))
        reconstruction_error = 1.0 / X_tilda.shape[0] * np.sum(
            np.sqrt(np.sum(np.multiply(X - X_tilda, X - X_tilda), axis=1)))

        return cross_entropy, reconstruction_error


if __name__ == "__main__":
    def plot_curve(label=""):
        # leading_label = sys.argv[0].split(".")[0] + "_" + str(int(time.time()))
        leading_label = "_" + str(int(time.time()))

        pickle.dump(plot_epoch_ce_train, open("../dump/train" + leading_label + "_ce_" + label, "w"))
        pickle.dump(plot_epoch_ce_valid, open("../dump/valid" + leading_label + "_ce_" + label, "w"))
        # pickle.dump(plot_epoch_ce_test, open("../dump/test" + leading_label + "_ce_" + label, "w"))
        pickle.dump(plot_epoch_re_train, open("../dump/train" + leading_label + "_re_" + label, "w"))
        pickle.dump(plot_epoch_re_valid, open("../dump/valid" + leading_label + "_re_" + label, "w"))

        # pickle.dump(plot_epoch_cr_train, open("../dump/train" + leading_label + "_cr_" + label, "w"))
        # pickle.dump(plot_epoch_cr_valid, open("../dump/valid" + leading_label + "_cr_" + label, "w"))
        # pickle.dump(plot_epoch_cr_test, open("../dump/test" + leading_label + "_cr_" + label, "w"))

        pickle.dump(model, open("../dump/model" + leading_label + "_" + label, "w"))
        # pickle.dump(speed, open("../dump/_speed_" + leading_label + "_" + label, "w"))


    np.seterr(all='raise')

    parser = argparse.ArgumentParser(description='data, parameters, etc.')
    parser.add_argument('-train', type=str, help='training file path', default='../data/digitstrain.txt')
    parser.add_argument('-valid', type=str, help='validation file path', default='../data/digitsvalid.txt')
    parser.add_argument('-test', type=str, help="test file path", default="../data/digitstest.txt")
    parser.add_argument('-max_epoch', type=int, help="maximum epoch", default=250)

    parser.add_argument('-n_hidden', type=int, help="num of hidden units", default=100)
    parser.add_argument('-k', type=int, help="CD-k sampling", default=1)
    parser.add_argument('-lr', type=float, help="learning rate", default=0.01)
    parser.add_argument('-minibatch_size', type=int, help="minibatch_size", default=1)

    args = parser.parse_args()

    train_data = np.genfromtxt(args.train, delimiter=",")
    train_X = train_data[:, :-1]  # all input pixels are non-negative here
    train_Y = train_data[:, -1]

    valid_data = np.genfromtxt(args.valid, delimiter=",")
    valid_X = valid_data[:, :-1]
    valid_Y = valid_data[:, -1]

    test_data = np.genfromtxt(args.test, delimiter=",")
    test_X = test_data[:, :-1]
    test_Y = test_data[:, -1]

    n_visible = train_X.shape[1]

    print "input dimension is " + str(n_visible)

    old_cross_entropy = sys.maxint
    old_classification_error = sys.maxint

    plot_epoch_ce_train = []
    plot_epoch_ce_valid = []
    plot_epoch_re_train = []
    plot_epoch_re_valid = []

    model = Model(n_visible, args.n_hidden, args.k, args.lr, args.minibatch_size)

    try:
        for epoch in range(args.max_epoch):
            random.seed(int(time.time()))
            train_X, train_Y = unison_shuffled_copies(train_X, train_Y)  # jointly shuffle the training data

            # model.store_old_parameter()
            print "epoch " + str(epoch) + " start "
            # then = time.time()

            if args.minibatch_size == 1:
                for instance_id in range(train_X.shape[0]):
                    if (instance_id % 1000 == 0):
                        print "sgd instance id " + str(instance_id)
                    model.update(train_X[instance_id])
            else:
                for instance_id in range(0, train_X.shape[0], args.minibatch_size)[:-1]:
                    model.update_minibatch(train_X[instance_id:min(instance_id + args.minibatch_size, len(train_X))])

            print "evaluating valid current learning rate " + str(model.lr)
            cross_entropy, reconstruction_error = model.eval(valid_X)

            # then = time.time()
            # cross_entropy, classification_error = model.eval_valid(valid_X, valid_Y, args.activation,
            #                                                        args.minibatch_size)
            # if old_classification_error < cross_entropy or old_classification_error < classification_error:
            plot_epoch_ce_valid += [cross_entropy]
            plot_epoch_re_valid += [reconstruction_error]
            # plot_epoch_cr_valid += [classification_error]
            print "cross_entropy valid reconstruction error " + str(cross_entropy) + " " + str(reconstruction_error)
            # + " " + str(classification_error)

            cross_entropy, reconstruction_error = model.eval(train_X)
            # , args.activation,
            #                                                    args.minibatch_size)
            plot_epoch_ce_train += [cross_entropy]
            plot_epoch_re_train += [reconstruction_error]
            # plot_epoch_cr_train += [classification_error]
            print "cross_entropy train reconstruction error " + str(cross_entropy) + " " + str(reconstruction_error)
            # + " " + str(classification_error)

            print


    # except KeyboardInterrupt as e:
    #     plot_curve()
    finally:
        plot_curve("_".join([name.strip("-") for name in sys.argv[1:]]))
