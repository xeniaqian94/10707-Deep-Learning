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


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class Model:
    def __init__(self, n_visible, n_hidden, lr, lbd, denoise):
        self.W = [[]]
        self.L_num_layer = 1
        self.b = [[]]
        self.lr = lr
        self.lbd = lbd
        self.denoise = denoise

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        dimension_list = [0, self.n_hidden, self.n_visible]

        np.random.seed(int(time.time()))

        self.W += [
            np.random.normal(0, 0.1, (self.n_hidden, self.n_visible))]
        self.W += [self.W[1].T]
        for i in range(1, self.L_num_layer + 2):
            self.b += [np.random.normal(0, 1e-10, dimension_list[i])]

    def forward(self, x, activation=sigmoid):
        h = [x]
        a = [[]]

        for i in range(1, self.L_num_layer + 2):
            a += [np.dot(self.W[i], h[i - 1]) + self.b[i]]
            h += [activation(a[i])]  # h_k=g(a_k)
            # print "current h_" + str(i) + " shape " + str(h[-1].shape)
        x_hat = h[-1]
        return a, h, x_hat

    def forward_batch(self, X, activation=sigmoid):
        h = [X]
        a = [[]]

        #
        # np.dot(self.W, v.T).T + np.tile(self.hbias, [v.shape[0], 1])
        #

        for i in range(1, self.L_num_layer + 2):
            a += [np.dot(self.W[i], h[i - 1].T).T + np.tile(self.b[i], [h[i - 1].shape[0], 1])]
            h += [activation(a[i])]  # h_k=g(a_k)
            # print "current h_" + str(i) + " shape " + str(h[-1].shape)
        x_hat = h[-1]
        return a, h, x_hat

    def back_prop(self, x, a, h, x_hat, deactivation=derivative_sigmoid):

        gradient_a = [None] * (self.L_num_layer + 2)
        gradient_W = [None] * (self.L_num_layer + 2)
        gradient_b = [None] * (self.L_num_layer + 2)
        gradient_h = [None] * (self.L_num_layer + 1)

        gradient_a[self.L_num_layer + 1] = x_hat - x
        # raw_input("gradient_a \n"+str(gradient_a[self.L_num_layer + 1]))

        for k in range(self.L_num_layer + 1, 0, -1):
            gradient_W[k] = np.outer(gradient_a[k], h[k - 1])
            # raw_input(str(k)+" gradient_W[k] h[k-1] \n" + str(gradient_W[k][0])+"\n"+str(h[k-1]))
            gradient_b[k] = gradient_a[k]

            if k == 1:
                break

            gradient_h[k - 1] = np.dot(self.W[k].T, gradient_a[k])
            gradient_a[k - 1] = np.multiply(gradient_h[k - 1], deactivation(h[k - 1], a[k - 1]))

        for i in range(1, self.L_num_layer + 2):
            delta_b_i = -1 * gradient_b[i] - 2.0 * self.lbd * self.b[i]

            self.b[i] = self.b[i] + self.lr * delta_b_i

        delta_W = -1.0 * (gradient_W[1] + gradient_W[2].T) - 2.0 * self.lbd * self.W[1]
        self.W[1] = self.W[1] + self.lr * delta_W
        self.W[2] = self.W[1].T

    def add_noise(self,x):
        percentage=int(self.denoise)*1.0/100

        return np.random.binomial(1,1-percentage,x.shape)*x

    def update(self, x):
        if len(self.denoise) > 0:
            a, h, x_hat = self.forward(self.add_noise(x))
        else:
            a, h, x_hat = self.forward(x)
        self.back_prop(x, a, h, x_hat)

    def reconstruction(self, v):
        # raw_input(v.shape)


        prob_h = self.prob_h_given_v(v)
        # raw_input(prob_h.shape)

        prob_v = self.prob_v_given_h(prob_h)
        # raw_input(prob_v.shape)
        return prob_v

    def eval(self, X):

        a, h, X_tilda = self.forward_batch(X)
        cross_entropy = -1.0 / X_tilda.shape[0] * np.sum(X * np.log(X_tilda) + (1 - X) * np.log(1 - X_tilda))
        reconstruction_error = 1.0 / X_tilda.shape[0] * np.sum(
            np.sqrt(np.sum(np.multiply(X - X_tilda, X - X_tilda), axis=1)))

        return cross_entropy, reconstruction_error


if __name__ == "__main__":
    def plot_curve(label=""):
        leading_label = "_" + str(int(time.time()))
        pickle.dump(plot_epoch_ce_train, open("../dump/autoencoder_train" + leading_label + "_ce_" + label, "w"))
        pickle.dump(plot_epoch_ce_valid, open("../dump/autoencoder_valid" + leading_label + "_ce_" + label, "w"))

        pickle.dump(plot_epoch_re_train, open("../dump/autoencoder_train" + leading_label + "_re_" + label, "w"))
        pickle.dump(plot_epoch_re_valid, open("../dump/autoencoder_valid" + leading_label + "_re_" + label, "w"))
        pickle.dump(model, open("../dump/autoencoder_model" + leading_label + "_" + label, "w"))


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

    parser.add_argument('-lbd', type=float, help="regularization term", default=0.000)
    parser.add_argument('-momentum', type=float, help="average gradient", default=0.5)
    parser.add_argument('-activation', type=str, help="which activation to use ", default="sigmoid")
    parser.add_argument('-denoise', type=str, default="")

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

    model = Model(n_visible, args.n_hidden, args.lr, args.lbd, args.denoise)

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
