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
    def __init__(self, L_num_layer, dimension_list, lr, lbd,
                 momentum, batch_normalization):  # here len(dimension_list) should equal to num_layer+2
        assert L_num_layer + 2 == len(dimension_list)
        self.L_num_layer = L_num_layer  # num_layer=L
        self.W = [[]]
        self.b = [[]]
        self.lr = lr
        self.new_W = []
        self.new_b = []
        self.sum_gradient_W = []
        self.sum_gradient_b = []
        self.old_W = []
        self.old_b = []
        self.lbd = lbd
        self.W_epsilon = [[]]
        self.prev_W = None
        self.prev_b = None
        self.momentum = momentum
        self.gamma = [[]]
        self.prev_gamma = [[]]
        self.beta = [[]]
        self.prev_beta = [[]]
        self.global_mean = [[]]
        self.global_variance = [[]]
        self.batch_normalization = batch_normalization

        self.num_class = dimension_list[-1]

        for i in range(1, self.L_num_layer + 2):
            self.W += [np.random.normal(0, 0.17, (dimension_list[i], dimension_list[i - 1]))]
            self.W_epsilon += [np.random.normal(0, 1e-5, (dimension_list[i], dimension_list[i - 1]))]
            self.b += [np.random.normal(0, 1e-10, dimension_list[i])]
            # print self.gamma
            self.gamma += [np.random.normal(0, 1, dimension_list[i])]
            self.beta += [np.random.normal(0, 1e-10, dimension_list[i])]

            self.global_mean += [np.random.normal(0, 1e-10, dimension_list[i])]
            self.global_variance += [np.random.normal(0, 1, dimension_list[i])]
            # self.gradient_W=np.zeros(self.W.shape)
            # self.gradient_b=np.zeros(self.b.shape)
            self.new_W = [None] * len(self.W)
            self.new_b = [None] * len(self.b)
            self.sum_gradient_W = [None] * len(self.W)
            self.sum_gradient_b = [None] * len(self.b)

            # self.W1=np.random.normal(0,1,(args.hidden_layer_1_dimension,num_dimension))
            # self.b1=np.random.normal(0,1,args.hidden_layer_1_dimension)
            # self.W2=np.random.normal(0,1,(args.num_class,args.hidden_layer_1_dimension))
            # self.b2=np.random.normal(0,1,args.num_class)

    def forward(self, x, activation=sigmoid):
        h = [x]
        a = [[]]

        for i in range(1, self.L_num_layer + 2):
            a += [np.dot(self.W[i], h[i - 1]) + self.b[i]]
            # print "current a_" + str(i) + " shape " + str(a[-1].shape)
            if i == self.L_num_layer + 1:
                break
            h += [activation(a[i])]  # h_k=g(a_k)
            # print "current h_" + str(i) + " shape " + str(h[-1].shape)
        # print "final a_" + str(i) + " shape " + str(a[-1].shape)
        softmax_over_class = softmax(a[-1])
        # print "softmax value " + str(softmax_over_class) + " " + str(np.sum(softmax_over_class))
        return softmax_over_class, a, h

    def forward_gradient_check(self, x, W_epsilon, activation=sigmoid):
        h = [x]
        a = [[]]

        for i in range(1, self.L_num_layer + 2):
            a += [np.dot(W_epsilon[i], h[i - 1]) + self.b[i]]
            # print "current a_" + str(i) + " shape " + str(a[-1].shape)
            if i == self.L_num_layer + 1:
                break
            h += [sigmoid(a[i])]  # h_k=g(a_k)
            # print "current h_" + str(i) + " shape " + str(h[-1].shape)
        # print "final a_" + str(i) + " shape " + str(a[-1].shape)
        softmax_over_class = softmax(a[-1])
        # print "softmax value " + str(softmax_over_class) + " " + str(np.sum(softmax_over_class))
        return softmax_over_class, a, h

    def gradient_check(self, i, delta_Wi, x, y):
        new_W_plus_epsilon = self.W
        new_W_plus_epsilon[i] += self.W_epsilon[i]
        f_W_plus_epsilon, a, h = self.forward_gradient_check(x, new_W_plus_epsilon)

        new_W_minus_epsilon = self.W
        new_W_minus_epsilon[i] -= self.W_epsilon[i]
        f_W_minus_epsilon, a, h = self.forward_gradient_check(x, new_W_minus_epsilon)

        print str(delta_Wi.shape) + " " + str(self.W_epsilon[i].shape) + " sanity check shape " + str(
            (f_W_plus_epsilon - f_W_minus_epsilon).shape)

    def back_prop(self, x, y, f_x, a, h, deactivation=derivative_sigmoid):
        e_y = np.zeros(self.num_class)
        e_y[y] = 1  # one hot class label

        gradient_a = [None] * (self.L_num_layer + 2)
        gradient_W = [None] * (self.L_num_layer + 2)
        gradient_b = [None] * (self.L_num_layer + 2)
        gradient_h = [None] * (self.L_num_layer + 1)

        gradient_a[self.L_num_layer + 1] = -1 * (e_y - f_x)
        # raw_input("gradient_a \n"+str(gradient_a[self.L_num_layer + 1]))

        for k in range(self.L_num_layer + 1, 0, -1):
            gradient_W[k] = np.outer(gradient_a[k], h[k - 1])
            # raw_input(str(k)+" gradient_W[k] h[k-1] \n" + str(gradient_W[k][0])+"\n"+str(h[k-1]))
            gradient_b[k] = gradient_a[k]

            if k == 1:
                break

            gradient_h[k - 1] = np.dot(self.W[k].T, gradient_a[k])
            # raw_input(str(k)+" gradient_h[k - 1] \n" + str(gradient_h[k - 1]))

            gradient_a[k - 1] = np.multiply(gradient_h[k - 1], deactivation(h[k - 1], a[k - 1]))
            # raw_input(str(k)+" gradient_a[k - 1] \n"+str(gradient_a[k - 1]))

        # eliminate the regularization term here
        # print gradient_b

        for i in range(1, self.L_num_layer + 2):
            delta_W_i = -1.0 * gradient_W[i] - 2.0 * self.lbd * self.W[i]
            delta_b_i = -1 * gradient_b[i] - 2.0 * self.lbd * self.b[i]

            if not self.prev_W == None and self.momentum > 0:
                delta_W_i -= 1.0 * self.momentum * self.prev_W[i]
                delta_b_i -= 1.0 * self.momentum * self.prev_b[i]

            self.W[i] = self.W[i] + self.lr * delta_W_i
            self.b[i] = self.b[i] + self.lr * delta_b_i

        self.prev_W = gradient_W
        self.prev_b = gradient_b

    def update(self, x, y, activation="sigmoid"):  # y is the final labely
        if activation == "sigmoid":
            f_x, a, h = self.forward(x)
            # print f_x.shape
            self.back_prop(x, y, f_x, a, h)
        elif activation == "tanh":
            f_x, a, h = self.forward(x, tanh)
            self.back_prop(x, y, f_x, a, h, deact_tanh)
        elif activation == "ReLU":
            f_x, a, h = self.forward(x, ReLU)
            self.back_prop(x, y, f_x, a, h, deact_ReLU)

    def forward_minibatch(self, X, activation=sigmoid):

        # print "within forward X.shape "+str(X.shape)

        h = [X.T]
        a = [[]]
        variable_cache = [[]]
        out = [[]]

        for i in range(1, self.L_num_layer + 2):

            a += [np.dot(self.W[i], h[i - 1]) + np.tile(self.b[i].reshape(self.b[i].shape[0], 1),
                                                        (1, h[i - 1].shape[
                                                            -1]))]  # resulting a is #hidden unit * minibatch size

            if self.batch_normalization:
                X = a[-1].T

                mu = np.mean(X, axis=0)  # (dimension,)
                # print "mu shape " + str(mu.shape)
                var = np.var(X, axis=0)

                X_norm = (X - mu) / np.sqrt(var + 1e-8)
                # print "X_norm shape " + str(X_norm.shape)
                # print "self.gamma[i] " + str(self.gamma[i].shape)
                # print "np.tile(self.gamma[i],(1,X_norm.shape[-1])) shape " + str(
                #     np.tile(self.gamma[i], (X_norm.shape[0], 1)).shape)
                # print "np.tile(self.beta[i],(1,X_norm.shape[-1])) shape " + str(
                #     np.tile(self.beta[i], (X_norm.shape[0], 1)).shape)

                out += [
                    np.tile(self.gamma[i], (X_norm.shape[0], 1)) * X_norm + np.tile(self.beta[i], (X_norm.shape[0], 1))]

                variable_cache += [(a[-1], X_norm, mu, var, self.gamma[i], self.beta[i])]

                self.global_mean[i] = 0.9 * self.global_mean[i] + 0.1 * mu
                self.global_variance[i] = 0.9 * self.global_variance[i] + 0.1 * var

                if i == self.L_num_layer + 1:
                    break

                h += [activation(out[-1].T)]  # h_k=g(a_k)
            else:
                if i==self.L_num_layer+1:
                    break
                h+=[activation(a[-1].T)]

        if self.batch_normalization:
            softmax_over_class = softmax(out[-1].T).T
        else:
            softmax_over_class = softmax(a[-1].T).T

        return softmax_over_class, a, h, out, variable_cache

    def back_prop_minibatch(self, X, Y, f_X, a, h, out, variable_cache, deactivation=derivative_sigmoid):
        # print "within back_prop"
        num_instance = X.shape[0]

        e_y = np.zeros([num_instance, self.num_class])
        # raw_input(type(list(Y)))

        # print list(range(len(Y))), list(Y)
        e_y[list(range(len(Y))), list(Y)] = 1  # one hot class label

        gradient_a = [None] * (
            self.L_num_layer + 2)  # except each is now not arraysize (num_class,)  but (num_instance,num_class)
        gradient_W = [None] * (self.L_num_layer + 2)
        gradient_b = [None] * (self.L_num_layer + 2)
        gradient_h = [None] * (self.L_num_layer + 1)
        gradient_out = [None] * (self.L_num_layer + 2)

        gradient_gamma = [None] * (self.L_num_layer + 2)
        gradient_beta = [None] * (self.L_num_layer + 2)

        # gradient_a[self.L_num_layer + 1] = -1 * (e_y - f_X)  # num_instance * num_class
        #
        # for k in range(self.L_num_layer + 1, 0, -1):
        #     gradient_W[k] = np.einsum('ki,jk->kij', gradient_a[k], h[k - 1])
        #     gradient_b[k] = gradient_a[k]
        #     gradient_h[k - 1] = np.dot(gradient_a[k], self.W[k]).T
        #     gradient_a[k - 1] = np.multiply(gradient_h[k - 1], deactivation(h[k - 1], a[k - 1])).T

        gradient_out[self.L_num_layer + 1] = -1 * (e_y - f_X)  # num_instance * num_class
        gradient_a[self.L_num_layer + 1] = -1 * (e_y - f_X)

        for k in range(self.L_num_layer + 1, 0, -1):

            if self.batch_normalization:
                ak, X_norm, mu, var, this_gamma, this_beta = variable_cache[k]

                X = ak.T

                N, D = X.shape

                X_mu = X - mu

                std_inv = 1. / np.sqrt(var + 1e-8)

                danorm = gradient_out[k] * this_gamma

                dvar = np.sum(danorm * X_mu, axis=0) * -.5 * std_inv ** 3

                dmu = np.sum(danorm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

                gradient_a[k] = ((danorm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)).T
                gradient_gamma[k] = np.sum(gradient_out[k] * X_norm, axis=0)
                gradient_beta[k] = np.sum(gradient_out[k], axis=0)

            gradient_W[k] = np.einsum('ik,jk->kij', gradient_a[k], h[k - 1])

            gradient_b[k] = gradient_a[k]

            if k==1:
                break
                
            gradient_h[k - 1] = np.dot(self.W[k].T, gradient_a[k])
            if self.batch_normalization:
                # print h[k - 1].shape, gradient_h[k - 1].shape
                # print out[k-1],h[k-1].shape
                gradient_out[k - 1] = np.multiply(gradient_h[k - 1], deactivation(h[k - 1], out[k - 1].T)).T
            else:
                gradient_a[k-1]=np.multiply(gradient_h[k - 1], deactivation(h[k - 1], a[k - 1])).T

        self.prev_W = [[]]
        self.prev_b = [[]]

        for i in range(1, self.L_num_layer + 2):
            delta_W_i = -1.0 * np.mean(gradient_W[i], axis=0) - 2.0 * self.lbd * self.W[i]
            # print "gradient b [i] shape "+str(gradient_b[i].shape)
            delta_b_i = -1 * np.mean(gradient_b[i], axis=1) - 2.0 * self.lbd * self.b[i]
            # print "gradient_gamma[i] " + str(gradient_gamma[i].shape)

            if self.batch_normalization:
                delta_gamma_i = -1.0 * gradient_gamma[i]
                delta_beta_i = -1.0 * gradient_beta[i]

            if len(self.prev_W) == len(self.W) and self.momentum > 0:
                delta_W_i -= 1.0 * self.momentum * self.prev_W[i]
                delta_b_i -= 1.0 * self.momentum * self.prev_b[i]
                if self.batch_normalization:
                    delta_gamma_i -= 1.0 * self.momentum * self.prev_gamma[i]
                    delta_beta_i -= 1.0 * self.momentum * self.prev_beta[i]

            self.W[i] = self.W[i] + self.lr * delta_W_i
            self.b[i] = self.b[i] + self.lr * delta_b_i
            if self.batch_normalization:
                self.gamma[i] = self.gamma[i] + self.lr * delta_gamma_i
                self.beta[i] = self.beta[i] + self.lr * delta_beta_i

            self.prev_W += [np.mean(gradient_W[i], axis=0)]
            self.prev_b += [np.mean(gradient_b[i], axis=1)]

        if self.batch_normalization:
            self.prev_gamma = gradient_gamma
            self.prev_beta = gradient_beta

    def update_minibatch(self, X, Y, activation_str="sigmoid"):  # y is the final labely
        if activation_str == "sigmoid":
            f_X, a, h, out, variable_cache = self.forward_minibatch(X, sigmoid)
            self.back_prop_minibatch(X, Y, f_X, a, h, out, variable_cache, derivative_sigmoid)
        elif activation_str == "tanh":
            f_X, a, h, out, variable_cache = self.forward_minibatch(X, tanh)
            self.back_prop_minibatch(X, Y, f_X, a, h, out, variable_cache, deact_tanh)
        elif activation_str == "ReLU":
            f_X, a, h, out, variable_cache = self.forward_minibatch(X, ReLU)
            self.back_prop_minibatch(X, Y, f_X, a, h, out, variable_cache, deact_ReLU)

    def reset_sum_gradient(self):
        self.sum_gradient_W = [None] * len(self.W)
        self.sum_gradient_b = [None] * len(self.b)

    def store_old_parameter(self):
        self.old_W = self.W
        self.old_b = self.b

    def eval_valid(self, valid_X, valid_Y, activation_str=sigmoid, minibatch=1):

        if activation_str == "sigmoid":
            activation = sigmoid
        elif activation_str == "ReLU":
            activation = ReLU
        elif activation_str == "tanh":
            activation = tanh

        h = [valid_X.T]
        a = [[]]
        out = [[]]

        for i in range(1, self.L_num_layer + 2):
            # print np.dot(self.W[i], h[i - 1]).shape
            # print np.tile(self.b[i].reshape(self.b[i].shape[0], 1), (1, h[i - 1].shape[1])).shape
            a += [
                np.dot(self.W[i], h[i - 1]) + np.tile(self.b[i].reshape(self.b[i].shape[0], 1), (1, h[i - 1].shape[1]))]
            # print "current a_" + str(i) + " shape " + str(a[-1].shape)

            if minibatch > 1 and self.batch_normalization:

                X = a[-1].T

                mu = np.mean(X, axis=0)  # (dimension,)

                var = np.var(X, axis=0)

                X_norm = (X - self.global_mean[i]) / np.sqrt(self.global_variance[i] + 1e-8)

                out += [
                    np.tile(self.gamma[i], (X_norm.shape[0], 1)) * X_norm + np.tile(self.beta[i], (X_norm.shape[0], 1))]

                if i == self.L_num_layer + 1:
                    break

                h += [activation(out[-1].T)]  # h_k=g(a_k)

        if minibatch > 1 and self.batch_normalization:
            softmax_over_class = softmax(out[-1].T).T
        else:
            softmax_over_class = softmax(a[-1]).T

        one_hot_y = self.from_Y_onehot(valid_Y)

        # max_prob_class=np.sum(softmax_over_class, axis=1)
        classification_error = np.sum(np.argmax(softmax_over_class, axis=1) != (np.asarray(valid_Y))) * 1.0 / len(
            valid_Y)

        return -1.0 * np.sum(np.multiply(np.log(softmax_over_class), one_hot_y)) / softmax_over_class.shape[
            0], classification_error

    def from_Y_onehot(self, valid_Y):
        row = range(len(valid_Y))
        col = np.asarray(valid_Y, dtype=int)
        one_hot_y = np.zeros([len(valid_Y), self.num_class])
        one_hot_y[row, col] = 1
        return one_hot_y

        # def store_global_mean_variance(self, trainX):


if __name__ == "__main__":
    def plot_curve(label=""):
        # leading_label = sys.argv[0].split(".")[0] + "_" + str(int(time.time()))
        leading_label = "_" + str(int(time.time()))

        # plt.plot(np.arange(len(plot_epoch_ce_train)), np.asarray(plot_epoch_ce_train), label="train")
        # plt.plot(np.arange(len(plot_epoch_ce_valid)), np.asarray(plot_epoch_ce_valid), label="valid")
        # plt.plot(np.arange(len(plot_epoch_ce_test)), np.asarray(plot_epoch_ce_test), label="test")
        #
        # plt.title("6(a) Average Training Cross-entropy Error vs. Epoch")
        # plt.xlabel("# of Epoches")
        # plt.ylabel("Avg Training Cross-entropy Error")
        # plt.grid()
        # plt.legend()
        # plt.show()
        # plt.savefig("../plot/cross_entropy_" + str(int(time.time())) + ".png")

        pickle.dump(plot_epoch_ce_train, open("../dump/train" + leading_label + "_ce_" + label, "w"))
        pickle.dump(plot_epoch_ce_valid, open("../dump/valid" + leading_label + "_ce_" + label, "w"))
        pickle.dump(plot_epoch_ce_test, open("../dump/test" + leading_label + "_ce_" + label, "w"))

        # plt.cla()

        # plt.plot(np.arange(len(plot_epoch_cr_train)), np.asarray(plot_epoch_cr_train) * 100, label="train")
        # plt.plot(np.arange(len(plot_epoch_cr_valid)), np.asarray(plot_epoch_cr_valid) * 100, label="valid")
        # plt.plot(np.arange(len(plot_epoch_cr_test)), np.asarray(plot_epoch_cr_test) * 100, label="test")
        # plt.title("6(b) Classification Error (in percentage vs. Epoch")
        # plt.xlabel("# of Epoches")
        # plt.ylabel("Classification Error (in percentage %)")
        #
        # plt.grid()
        # plt.legend()
        # plt.show()
        # plt.savefig("../plot/classification_error_" + str(int(time.time())) + ".png")

        pickle.dump(plot_epoch_cr_train, open("../dump/train" + leading_label + "_cr_" + label, "w"))
        pickle.dump(plot_epoch_cr_valid, open("../dump/valid" + leading_label + "_cr_" + label, "w"))
        pickle.dump(plot_epoch_cr_test, open("../dump/test" + leading_label + "_cr_" + label, "w"))

        pickle.dump(model, open("../dump/_model_" + leading_label + "_" + label, "w"))
        pickle.dump(speed, open("../dump/_speed_" + leading_label + "_" + label, "w"))


    np.seterr(all='raise')

    random.seed(int(time.time()))

    parser = argparse.ArgumentParser(description='data, parameters, etc.')
    parser.add_argument('-train', type=str, help='training file path', default='../data/digitstrain.txt')
    parser.add_argument('-valid', type=str, help='validation file path', default='../data/digitsvalid.txt')
    parser.add_argument('-test', type=str, help="test file path", default="../data/digitstest.txt")
    parser.add_argument('-max_epoch', type=int, help="maximum epoch", default=250)
    parser.add_argument('-num_layer', type=int, help="if 1 single layer, 2 then 2-layer network", default=1)
    parser.add_argument('-num_class', type=int, default=10)
    parser.add_argument('-hidden_layer_1_dimension', type=int, default=100)
    parser.add_argument('-hidden_layer_2_dimension', type=int, default=0)
    parser.add_argument('-lr', type=float, help="learning rate", default=0.01)
    parser.add_argument('-lbd', type=float, help="regularization term", default=0.001)
    parser.add_argument('-momentum', type=float, help="average gradient", default=0.5)
    parser.add_argument('-minibatch_size', type=int, help="minibatch_size", default=1)
    parser.add_argument('-batch_normalization', type=bool, help="whether do batch normalization or not ", default=True)
    parser.add_argument('-activation', type=str, help="which activation to use ", default="sigmoid")

    # python train.py -minibatch_size 32 -batch_normalization True


    args = parser.parse_args()

    train_data = np.genfromtxt(args.train, delimiter=",")
    train_X = train_data[:, :-1]
    train_Y = train_data[:, -1]
    # print train_Y[-20:-1]

    valid_data = np.genfromtxt(args.valid, delimiter=",")
    valid_X = valid_data[:, :-1]
    valid_Y = valid_data[:, -1]

    test_data = np.genfromtxt(args.test, delimiter=",")
    test_X = test_data[:, :-1]
    test_Y = test_data[:, -1]

    num_dimension = train_X.shape[1]

    old_cross_entropy = sys.maxint
    old_classification_error = sys.maxint

    plot_epoch_ce_train = []
    plot_epoch_ce_valid = []
    plot_epoch_ce_test = []

    plot_epoch_cr_train = []
    plot_epoch_cr_valid = []
    plot_epoch_cr_test = []
    speed = []
    model = Model(args.num_layer, [num_dimension, args.hidden_layer_1_dimension, args.num_class], args.lr, args.lbd,
                  args.momentum, args.batch_normalization)
    if not args.hidden_layer_2_dimension == 0:
        args.num_layer = 2
        model = Model(args.num_layer,
                      [num_dimension, args.hidden_layer_1_dimension, args.hidden_layer_2_dimension, args.num_class],
                      args.lr, args.lbd,
                      args.momentum)

    try:
        for epoch in range(args.max_epoch):

            train_X, train_Y = unison_shuffled_copies(train_X, train_Y)

            model.store_old_parameter()
            print "epoch " + str(epoch) + " start "
            then = time.time()

            if args.minibatch_size == 1:
                for instance_id in range(train_X.shape[0]):
                    if (instance_id % 1000 == 0):
                        print "sgd instance id " + str(instance_id)
                    model.update(train_X[instance_id], train_Y[instance_id], args.activation)
            else:
                for instance_id in range(0, train_X.shape[0], args.minibatch_size)[:-1]:
                    model.update_minibatch(train_X[instance_id:min(instance_id + args.minibatch_size, len(train_X))],
                                           train_Y[instance_id:min(instance_id + args.minibatch_size, len(train_Y))],
                                           args.activation)
            speed_in_secs = time.time() - then
            print "epoch " + str(epoch) + " end in " + str(speed_in_secs)
            speed += [speed_in_secs]

            print "evaluating valid current learning rate " + str(model.lr)
            then = time.time()
            cross_entropy, classification_error = model.eval_valid(valid_X, valid_Y, args.activation,
                                                                   args.minibatch_size)
            # if old_classification_error < cross_entropy or old_classification_error < classification_error:
            plot_epoch_ce_valid += [cross_entropy]
            plot_epoch_cr_valid += [classification_error]
            print "cross_entropy classification_error valid " + str(cross_entropy) + " " + str(classification_error)

            cross_entropy, classification_error = model.eval_valid(train_X, train_Y, args.activation,
                                                                   args.minibatch_size)
            plot_epoch_ce_train += [cross_entropy]
            plot_epoch_cr_train += [classification_error]
            print "cross_entropy classification_error train " + str(cross_entropy) + " " + str(classification_error)

            cross_entropy, classification_error = model.eval_valid(test_X, test_Y, args.activation, args.minibatch_size)
            plot_epoch_ce_test += [cross_entropy]
            plot_epoch_cr_test += [classification_error]
            print "cross_entropy classification_error test " + str(cross_entropy) + " " + str(classification_error)

            print "evaluating valid end in " + str(time.time() - then)

            # print model.W[1][:20, :20]

    # except KeyboardInterrupt as e:
    #     plot_curve()
    finally:
        plot_curve("_".join([name.strip("-") for name in sys.argv[1:]]))




        # train_Y
        # loop.close()

        # if epoch==1:
        #     break


# print model.W1.shape
#
#
# print train_X[0].shape
# print W1.shape
# print (np.dot(W1,train_X[0])+b).shape
# print b.shape

# for epoch in range(args.max_epoch):




# print X.shape
# print Y.shape
# print type(Y)
#
# x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
# y = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#
# row_id=int(random.random()*len(X))
# grid = X[row_id].reshape(np.sqrt(X.shape[1]),np.sqrt(X.shape[1]))
# print Y[row_id]
#
#
# plt.imshow(grid,cmap='gray')
# plt.show()
# plt.cla()
