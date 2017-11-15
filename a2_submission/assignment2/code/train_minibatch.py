import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time
import math
import argparse
import matplotlib.pyplot as plt
import pickle


def sigmoid(x):
    try:
        return 1 / (1 + np.exp(-x))
    except:
        print "exception sigmoid!"
        sys.exit(0)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    try:
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    except:
        print "exception softmax"
        print x
        sys.exit(0)


def derivative_sigmoid(g_a):
    return np.multiply(g_a, 1 - g_a)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class Model:
    def __init__(self, L_num_layer, dimension_list, lr, lbd,
                 minibatch_size):  # here len(dimension_list) should equal to num_layer+2
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
        self.minibatch_size = minibatch_size

        print dimension_list
        self.num_class = dimension_list[-1]

        for i in range(1, self.L_num_layer + 2):
            self.W += [np.random.normal(0, 1, (dimension_list[i], dimension_list[i - 1]))]
            self.W_epsilon += [np.random.normal(0, 1e-5, (dimension_list[i], dimension_list[i - 1]))]
            self.b += [np.random.normal(0, 1, dimension_list[i])]
            # self.gradient_W=np.zeros(self.W.shape)
            # self.gradient_b=np.zeros(self.b.shape)
            self.new_W = [None] * len(self.W)
            self.new_b = [None] * len(self.b)
            self.sum_gradient_W = [None] * len(self.W)
            self.sum_gradient_b = [None] * len(self.b)

    def forward(self, X, activation=sigmoid):

        # print "within forward X.shape "+str(X.shape)

        h = [X.T]
        a = [[]]


        for i in range(1, self.L_num_layer + 2):

            # raw_input("np.dot(self.W[i], h[i - 1]) + np.tile(self.b[i].reshape(self.b[i].shape[0], 1), (1, h[i - 1].shape[1])\n"+str(self.W[i].shape)+" "+str(h[i-1].shape))
            # raw_input("self.b[i].reshape "+str(self.b[i].shape)+" "+str(h[i - 1].shape[-1]))
            a += [np.dot(self.W[i], h[i - 1]) + np.tile(self.b[i].reshape(self.b[i].shape[0], 1), (1, h[i - 1].shape[-1]))]
            # print "current a_" + str(i) + " shape " + str(a[-1].shape)
            if i == self.L_num_layer + 1:
                break
            h += [sigmoid(a[i])]  # h_k=g(a_k)
            # print "current h_" + str(i) + " shape " + str(h[-1].shape)
        # print "final a_" + str(i) + " shape " + str(a[-1].shape)
        softmax_over_class = softmax(a[-1]).T   # size num_instances * num_class
        # print "softmax value " + str(softmax_over_class) + " " + str(np.sum(softmax_over_class))

        return softmax_over_class, a, h

    def back_prop(self, X, Y, f_X, a, h):
        # print "within back_prop"
        num_instance=X.shape[0]

        e_y = np.zeros([num_instance,self.num_class])
        # raw_input(type(list(Y)))

        # print list(range(len(Y))), list(Y)
        e_y[list(range(len(Y))), list(Y)] = 1  # one hot class label

        gradient_a = [None] * (self.L_num_layer + 2)  # except each is now not arraysize (num_class,)  but (num_instance,num_class)
        gradient_W = [None] * (self.L_num_layer + 2)
        gradient_b = [None] * (self.L_num_layer + 2)
        gradient_h = [None] * (self.L_num_layer + 1)

        gradient_a[self.L_num_layer + 1] = -1 * (e_y - f_X)  #num_instance * num_class

        # raw_input("gradient_a \n"+str(gradient_a[self.L_num_layer + 1]))

        for k in range(self.L_num_layer + 1, 0, -1):

            # raw_input(str(k)+" gradient_a[k] h[k-1].T dot product \n" + str(gradient_a[k].shape)+" "+str(h[k-1].shape))

            # gradient_W[k]=np.dot(gradient_a[k],h[k-1].T)
            gradient_W[k]=np.einsum('ki,jk->kij',gradient_a[k],h[k-1])

            gradient_b[k] = gradient_a[k]

            # raw_input(
            #     str(k) + "  self.W[k]  gradient_a[k] \n" + str(self.W[k].shape) + " " + str(gradient_a[k].shape))

            gradient_h[k - 1] = np.dot(gradient_a[k],self.W[k]).T



            # raw_input(str(k)+" np.multiply(gradient_h[k - 1], derivative_sigmoid(h[k - 1]) " + str(gradient_h[k - 1].shape)+" "+str(h[k-1].shape))

            gradient_a[k - 1] = np.multiply(gradient_h[k - 1], derivative_sigmoid(h[k - 1])).T
            # raw_input(str(k)+" gradient_a[k - 1] \n"+str(gradient_a[k - 1]))

        # eliminate the regularization term here
        # print gradient_b

        for i in range(1, self.L_num_layer + 2):

            # raw_input("delta_W_i = -1.0 * gradient_W[i] - 2.0 * self.lbd * self.W[i] "+str(gradient_W[i].shape))
            delta_W_i = -1.0 * np.mean(gradient_W[i],axis=0) - 2.0 * self.lbd * self.W[i]
            # delta_W_i = -1.0 * gradient_W[i]

            # self.gradient_check(i,gradient_W[i],x,y)

            # self.new_W[i] = self.W[i] + self.lr * delta_W_i
            self.W[i] = self.W[i] + self.lr * delta_W_i
            # if self.sum_gradient_W[i] ==None:
            #     self.sum_gradient_W[i]=delta_W_i
            # else:
            #     self.sum_gradient_W[i] +=delta_W_i

            delta_b_i = -1 * np.mean(gradient_b[i],axis=0) - 2.0 * self.lbd * self.b[i]

            # delta_b_i = -1.0 * gradient_b[i]

            # self.new_b[i] = self.b[i] + self.lr * delta_b_i
            self.b[i] = self.b[i] + self.lr * delta_b_i

            # print str(i)+" "+str(self.b[i].shape)

            # if self.sum_gradient_b[i] ==None:
            #     self.sum_gradient_b[i]=delta_b_i
            # else:
            #     self.sum_gradient_b[i] +=delta_b_i


            # print "weight updated here! are the new_W and W equal? "
            # print self.new_W[1]==self.W[1]
            # print (np.all(self.new_W[1]==self.W[1]))
            # print str(self.new_W[1][:3,:3])
            # print str(self.W[1][:3,:3])

    def update(self, X, Y):  # y is the final labely
        f_X, a, h = self.forward(X)
        self.back_prop(X, Y, f_X, a, h)

    def reset_sum_gradient(self):
        self.sum_gradient_W = [None] * len(self.W)
        self.sum_gradient_b = [None] * len(self.b)

    def store_old_parameter(self):
        self.old_W = self.W
        self.old_b = self.b

    def eval_valid(self, valid_X, valid_Y):
        h = [valid_X.T]
        a = [[]]

        for i in range(1, self.L_num_layer + 2):
            # print np.dot(self.W[i], h[i - 1]).shape
            # print np.tile(self.b[i].reshape(self.b[i].shape[0], 1), (1, h[i - 1].shape[1])).shape
            a += [
                np.dot(self.W[i], h[i - 1]) + np.tile(self.b[i].reshape(self.b[i].shape[0], 1), (1, h[i - 1].shape[1]))]
            # print "current a_" + str(i) + " shape " + str(a[-1].shape)
            if i == self.L_num_layer + 1:
                break
            h += [sigmoid(a[i])]  # h_k=g(a_k)
            # print "current h_" + str(i) + " shape " + str(h[-1].shape)
        # print "final a_" + str(i) + " shape " + str(a[-1].shape)

        softmax_over_class = softmax(a[-1]).T  # size num_instances * num_class
        # print "softmax value " + str(softmax_over_class) + " " + str(np.sum(softmax_over_class))

        one_hot_y = self.from_Y_onehot(valid_Y)

        # max_prob_class=np.sum(softmax_over_class, axis=1)
        classification_error = np.sum(np.argmax(softmax_over_class, axis=1) != (np.asarray(valid_Y))) * 1.0 / len(
            valid_Y)
        # print "classification error calculation "
        # print softmax_over_class[:5,:]
        # print valid_Y[:5]
        # print np.sum(np.argmax(softmax_over_class[:5,:], axis=1) != (np.asarray(valid_Y[:5]))) * 1.0 / len(
        #     valid_Y)

        return -1.0 * np.sum(np.multiply(np.log(softmax_over_class), one_hot_y)) / softmax_over_class.shape[
            0], classification_error

    def from_Y_onehot(self, valid_Y):
        row = range(len(valid_Y))
        col = np.asarray(valid_Y, dtype=int)
        one_hot_y = np.zeros([len(valid_Y), self.num_class])
        one_hot_y[row, col] = 1
        return one_hot_y


if __name__ == "__main__":
    def plot_curve(label=""):
        # time_label = str(int(time.time()))
        leading_label=sys.argv[0].split(".")[0]


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

        pickle.dump(plot_epoch_ce_train, open("../dump/"+leading_label + "_ce_" + label, "w"))
        pickle.dump(plot_epoch_ce_valid, open("../dump/"+leading_label + "_ce_" + label, "w"))
        pickle.dump(plot_epoch_ce_test, open("../dump/"+leading_label + "_ce_" + label, "w"))

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

        pickle.dump(plot_epoch_cr_train, open("../dump/"+leading_label + "_cr_" + label, "w"))
        pickle.dump(plot_epoch_cr_valid, open("../dump/"+leading_label + "_cr_" + label, "w"))
        pickle.dump(plot_epoch_cr_test, open("../dump/"+leading_label + "_cr_" + label, "w"))
        pickle.dump(model, open("../dump/"+leading_label+ "_model_" + label, "w"))


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
    parser.add_argument('-lr', type=float, help="learning rate", default=0.1)
    parser.add_argument('-lbd', type=float, help="regularization term", default=0.001)
    parser.add_argument('-minibatch_size', type=int, help="minibatch size", default=128)

    args = parser.parse_args()

    train_data = np.genfromtxt(args.train, delimiter=",")
    train_X = train_data[:, :-1]
    train_Y = np.asarray(train_data[:, -1],dtype=int)
    # print train_Y[-20:-1]

    valid_data = np.genfromtxt(args.valid, delimiter=",")
    valid_X = valid_data[:, :-1]
    valid_Y = np.asarray(valid_data[:, -1],dtype=int)

    test_data = np.genfromtxt(args.test, delimiter=",")
    test_X = test_data[:, :-1]
    test_Y = np.asarray(test_data[:, -1],dtype=int)

    num_dimension = train_X.shape[1]

    old_cross_entropy = sys.maxint
    old_classification_error = sys.maxint

    plot_epoch_ce_train = []
    plot_epoch_ce_valid = []
    plot_epoch_ce_test = []

    plot_epoch_cr_train = []
    plot_epoch_cr_valid = []
    plot_epoch_cr_test = []
    model = Model(args.num_layer, [num_dimension, args.hidden_layer_1_dimension, args.num_class], args.lr, args.lbd,
                  args.minibatch_size)

    try:
        for epoch in range(args.max_epoch):

            train_X, train_Y = unison_shuffled_copies(train_X, train_Y)

            model.store_old_parameter()

            print "epoch " + str(epoch) + " start "
            then = time.time()

            for instance_id in range(0, train_X.shape[0], args.minibatch_size)[:-1]:
                model.update(train_X[instance_id:min(instance_id+args.minibatch_size,len(train_X))], train_Y[instance_id:min(instance_id+args.minibatch_size,len(train_Y))])

            print "epoch " + str(epoch) + " end in " + str(time.time() - then)

            print "evaluating valid current learning rate " + str(model.lr)
            then = time.time()
            cross_entropy, classification_error = model.eval_valid(valid_X, valid_Y)
            # if old_classification_error < cross_entropy or old_classification_error < classification_error:
            if epoch > 200:
                model.lr = 0.01
                # model.lr=model.lr*0.1
            if cross_entropy < old_cross_entropy:
                old_cross_entropy = cross_entropy
            else:
                model.W = model.old_W
                model.b = model.old_b

            plot_epoch_ce_valid += [cross_entropy]
            plot_epoch_cr_valid += [classification_error]
            print "cross_entropy classification_error valid " + str(cross_entropy) + " " + str(classification_error)

            cross_entropy, classification_error = model.eval_valid(train_X, train_Y)
            plot_epoch_ce_train += [cross_entropy]
            plot_epoch_cr_train += [classification_error]
            print "cross_entropy classification_error train " + str(cross_entropy) + " " + str(classification_error)

            cross_entropy, classification_error = model.eval_valid(test_X, test_Y)
            plot_epoch_ce_test += [cross_entropy]
            plot_epoch_cr_test += [classification_error]
            print "cross_entropy classification_error test " + str(cross_entropy) + " " + str(classification_error)

            print "evaluating valid end in " + str(time.time() - then)

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
