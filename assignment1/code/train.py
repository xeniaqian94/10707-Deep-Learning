import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time
import math
import argparse

np.seterr(all='raise')

random.seed(int(time.time()))

parser = argparse.ArgumentParser(description='data, parameters, etc.')
parser.add_argument('-train', type=str, help='training file path', default='../data/digitstrain.txt')
parser.add_argument('-valid', type=str, help='validation file path', default='../data/digitsvalid.txt')
parser.add_argument('-test',type=str,help="test file path",default="../data/digitstest.txt")
parser.add_argument('-max_epoch', type=int, help="maximum epoch", default=1000)
parser.add_argument('-num_layer', type=int, help="if 1 single layer, 2 then 2-layer network", default=1)
parser.add_argument('-num_class', type=int, default=10)
parser.add_argument('-hidden_layer_1_dimension', type=int, default=100)
parser.add_argument('-lr', type=float, help="learning rate", default=0.1)
parser.add_argument('-lbd', type=float, help="regularization term", default=0.001)
args = parser.parse_args()

train_data = np.genfromtxt(args.train, delimiter=",")
train_X = train_data[:, :-1]
train_Y = train_data[:, -1]
print train_Y[-20:-1]

valid_data = np.genfromtxt(args.valid, delimiter=",")
valid_X = valid_data[:, :-1]
valid_Y = valid_data[:, -1]

test_data = np.genfromtxt(args.test, delimiter=",")
test_X = test_data[:, :-1]
test_Y = test_data[:, -1]

num_dimension = train_X.shape[1]


# train_Y

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


class Model:
    def __init__(self, L_num_layer, dimension_list, lr, lbd):  # here len(dimension_list) should equal to num_layer+2
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
        self.W_epsilon=[[]]

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
            h += [sigmoid(a[i])]  # h_k=g(a_k)
            # print "current h_" + str(i) + " shape " + str(h[-1].shape)
        # print "final a_" + str(i) + " shape " + str(a[-1].shape)
        softmax_over_class = softmax(a[-1])
        # print "softmax value " + str(softmax_over_class) + " " + str(np.sum(softmax_over_class))
        return softmax_over_class, a, h

    def forward_gradient_check(self,x,W_epsilon,activation=sigmoid):
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

    def gradient_check(self,i,delta_Wi,x,y):
        new_W_plus_epsilon=self.W
        new_W_plus_epsilon[i]+=self.W_epsilon[i]
        f_W_plus_epsilon,a,h=self.forward_gradient_check(x,new_W_plus_epsilon)

        new_W_minus_epsilon = self.W
        new_W_minus_epsilon[i] -= self.W_epsilon[i]
        f_W_minus_epsilon,a,h = self.forward_gradient_check(x, new_W_minus_epsilon)

        print str(delta_Wi.shape)+" "+str(self.W_epsilon[i].shape)+" sanity check shape "+str((f_W_plus_epsilon-f_W_minus_epsilon).shape)





    def back_prop(self, x, y, f_x, a, h):
        e_y = np.zeros(self.num_class)
        e_y[y] = 1  # one hot class label

        delta_a = [None] * (self.L_num_layer + 2)
        delta_W = [None] * (self.L_num_layer + 2)
        delta_b = [None] * (self.L_num_layer + 2)
        delta_h = [None] * (self.L_num_layer + 1)

        delta_a[self.L_num_layer + 1] = -1 * (e_y - f_x)
        # raw_input("delta_a \n"+str(delta_a[self.L_num_layer + 1]))

        for k in range(self.L_num_layer + 1, 0, -1):
            delta_W[k] = np.outer(delta_a[k], h[k - 1])
            # raw_input(str(k)+" delta_W[k] h[k-1] \n" + str(delta_W[k][0])+"\n"+str(h[k-1]))
            delta_b[k] = delta_a[k]

            delta_h[k - 1] = np.dot(self.W[k].T, delta_a[k])
            # raw_input(str(k)+" delta_h[k - 1] \n" + str(delta_h[k - 1]))


            delta_a[k - 1] = np.multiply(delta_h[k - 1], derivative_sigmoid(h[k - 1]))
            # raw_input(str(k)+" delta_a[k - 1] \n"+str(delta_a[k - 1]))

        # eliminate the regularization term here
        # print delta_b

        for i in range(1, self.L_num_layer + 2):
            # gradient_W_i = -1.0 * delta_W[i] - 2.0 * self.lbd * self.W[i]
            gradient_W_i = -1.0 * delta_W[i]

            # self.gradient_check(i,delta_W[i],x,y)

            # self.new_W[i] = self.W[i] + self.lr * gradient_W_i
            self.W[i] = self.W[i] + self.lr * gradient_W_i
            # if self.sum_gradient_W[i] ==None:
            #     self.sum_gradient_W[i]=gradient_W_i
            # else:
            #     self.sum_gradient_W[i] +=gradient_W_i

            # gradient_b_i = -1 * delta_b[i] - 2.0 * self.lbd * self.b[i]

            gradient_b_i = -1.0 * delta_b[i]

            # self.new_b[i] = self.b[i] + self.lr * gradient_b_i
            self.b[i] = self.b[i] + self.lr * gradient_b_i

            # if self.sum_gradient_b[i] ==None:
            #     self.sum_gradient_b[i]=gradient_b_i
            # else:
            #     self.sum_gradient_b[i] +=gradient_b_i


            # print "weight updated here! are the new_W and W equal? "
            # print self.new_W[1]==self.W[1]
            # print (np.all(self.new_W[1]==self.W[1]))
            # print str(self.new_W[1][:3,:3])
            # print str(self.W[1][:3,:3])

    def update(self, x, y):  # y is the final labely
        f_x, a, h = self.forward(x)
        # print f_x.shape
        self.back_prop(x, y, f_x, a, h)

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
            print np.dot(self.W[i], h[i - 1]).shape
            print np.tile(self.b[i].reshape(self.b[i].shape[0], 1), (1, h[i - 1].shape[1])).shape
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

        #
        #
        # if self.L_num_layer==1:
        #
        #     h1=np.dot(self.W1,x)+self.b1
        #     print h1.shape #
        #     a1=sigmoid(h1)
        #     sigmoid(np.dot(self.W1,x)+self.b1)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


old_cross_entropy = sys.maxint
old_classification_error = sys.maxint

plot_epoch_train=[]
plot_epoch_valid=[]
plot_epoch_test=[]


import matplotlib.pyplot as plt


def plot_curve():
    plt.plot(np.arange(len(plot_epoch_train)),plot_epoch_train,label="train")
    plt.plot(np.arange(len(plot_epoch_valid)),plot_epoch_valid,label="valid")
    plt.plot(np.arange(len(plot_epoch_test)), plot_epoch_test,label="test")
    plt.legend()
    plt.show()

try:
    model = Model(args.num_layer, [num_dimension, args.hidden_layer_1_dimension, args.num_class], args.lr, args.lbd)
    for epoch in range(args.max_epoch):

        train_X, train_Y = unison_shuffled_copies(train_X, train_Y)

        model.store_old_parameter()
        print "epoch " + str(epoch) + " start "
        then = time.time()

        for instance_id in range(train_X.shape[0]):
            if (instance_id % 1000 == 0):
                print "sgd instance id " + str(instance_id)
            model.update(train_X[instance_id], train_Y[instance_id])
            # if instance_id==1:
            #     break

        print "epoch " + str(epoch) + " end in " + str(time.time() - then)

        print "evaluating valid current learning rate " + str(model.lr)
        then = time.time()
        cross_entropy, classification_error = model.eval_valid(valid_X, valid_Y)
        # if old_classification_error < cross_entropy or old_classification_error < classification_error:
        if epoch>15:
            model.lr=0.01
            # model.lr=model.lr*0.1
            pass
        else:
            old_cross_entropy = cross_entropy
            old_classification_error = classification_error
        plot_epoch_valid+=[cross_entropy]
        print "cross_entropy classification_error valid " + str(cross_entropy) + " " + str(classification_error)

        cross_entropy, classification_error = model.eval_valid(train_X, train_Y)
        plot_epoch_train+=[cross_entropy]
        print "cross_entropy classification_error train " + str(cross_entropy) + " " + str(classification_error)

        cross_entropy, classification_error = model.eval_valid(test_X, test_Y)
        plot_epoch_test+=[cross_entropy]
        print "cross_entropy classification_error test " + str(cross_entropy) + " " + str(classification_error)

        print "evaluating valid end in " + str(time.time() - then)


except KeyboardInterrupt as e:
    plot_curve()
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
