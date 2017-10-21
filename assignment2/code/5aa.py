import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
import pickle
import os
import re
from train import Model

# python 5aa.py ../dump/ 1508612534

dir_file_path = sys.argv[1]  # ../dump/plot_epoch_
files = os.listdir(dir_file_path)

candidate_model_filename = [file for file in files if sys.argv[2] in file and "model" in file][0]

model = pickle.load(open(dir_file_path + candidate_model_filename, "r"))

print "model W shape " + str((model.W.shape))

fig = plt.figure()

f, axarr = plt.subplots(int(np.sqrt(model.W.shape[0])), int(np.sqrt(model.W.shape[0])))
plt.gray()

for i in np.arange(int(model.W.shape[0])):
    row = i / 10
    col = i % 10
    this_array = np.asarray(model.W[i]).reshape(np.sqrt(model.W.shape[1]), np.sqrt(model.W.shape[1]))
    axarr[row, col].imshow(this_array)
    axarr[row, col].axis('off')

f.subplots_adjust()
# plt.show()
f.savefig("../plot/" + candidate_model_filename.split("/")[-1] + ".png")

plt.cla()

fig = plt.figure()
#
for (metric, metric_full_name) in [('ce', "Average Cross Entropy ")]:
    this_metric_fold = dict()
    for (color, fold) in [("b", 'train'), ("g", 'valid')]:
        candidate_file = \
        [filename for filename in files if fold in filename and metric in filename and sys.argv[2] in filename][0]
        this_list = np.asarray(pickle.load(open(dir_file_path + candidate_file, "r")))
        plt.plot(np.arange(len(this_list)), this_list, color, label=fold + " ")

    plt.title(metric_full_name + "vs. Epoch")
    plt.xlabel("# of Epoches")
    plt.ylabel(metric_full_name)

    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig("../plot/" + candidate_model_filename.split("/")[-1].strip("_")+"_" + metric + ".png")
    plt.cla()









#
#         # question 6d
#         # for (lr,line_style) in [(0.1,"-."),(0.01,":"),(0.2,"--"),(0.5,"-")]:
#         #     candidate_files = [filename for filename in files if re.match(".*lr_" + str(lr) + "$",filename) and fold in filename and metric in filename and str(lr) in filename]
#
#         # question 6d2
#         # for (lr,line_style) in [(0.0,"-."),(0.5,"--"),(0.9,"-")]:
#         #     candidate_files=[filename for filename in files if re.match(".*lr_0.01_momentum_" + str(lr) + "$",filename) and fold in filename and metric in filename and str(lr) in filename]
#
#         # question 6e
#         # for (lr,line_style) in [(20,"-."),(100,":"),(200,"--"),(500,"-")]:
#         #     candidate_files = [filename for filename in files if re.match(".*lr_0.01_momentum_0.5_hidden_layer_1_dimension_" + str(lr) + "$",filename) and fold in filename and metric in filename and str(lr) in filename]
#
#         # question 6h BN or not
#         # for (lr, line_style) in [("True", "-"), ("False", "--")]:
#         #     this_metric_fold[fold] = None
#         #     candidate_files = [filename for filename in files if re.match(
#         #         ".*minibatch_size_32_batch_normalization_" + str(lr) + "$",
#         #         filename) and fold in filename and metric in filename]
#         #     print len(candidate_files)
#
#             # question 6i
#             # for (lr, line_style) in [('tanh', "-."),('sigmoid', ":"), ('ReLU', "--")]:
#             #     candidate_files = [filename for filename in files if re.match(".*lr_0.01_momentum_0.5_hidden_layer_2_dimension_100_activation_" + str(lr) + "*",filename) and fold in filename and metric in filename]
#
#         # for (lr, line_style) in [("32", "-"),("64","--"),("128","-.")]:
#         #
#         #     candidate_files = [filename for filename in files if re.match(
#         #         ".*tanh_minibatch_size_"+str(lr)+"$",
#         #         filename) and fold in filename and metric in filename]
#
#         for (lr, line_style) in [("0.01", "-"),("0.1","--"),("0.2","-."),("0.5",":")]:
#             candidate_files = [filename for filename in files if re.match(
#                 ".*lr_"+lr+"_minibatch_size_32$",
#                 filename) and fold in filename and metric in filename]
#             this_metric_fold[fold] = None
#             print len(candidate_files)
#
#             count = 0
#             for ind, filename in enumerate(candidate_files):
#
#                 this_list = np.asarray(pickle.load(open(dir_file_path + filename, "r")))
#                 if metric == "cr":
#                     this_list = this_list * 100  # for percentage
#                 print str(filename), str(type(this_list))
#                 # if len(this_list) == 250:
#                 if not ind:
#                     this_metric_fold[fold] = this_list
#                     count += 1
#                 else:
#                     if not len(this_list) == len(this_metric_fold[fold]):
#                         print len(this_list), len(this_metric_fold[fold])
#                         this_metric_fold[fold] = this_metric_fold[fold][
#                                                  :min(len(this_list), len(this_metric_fold[fold]))]
#                     count += 1
#                     this_metric_fold[fold] += this_list
#
#                 print metric, fold, this_list[:10], this_list[-10:]
#
#             this_metric_fold[fold] /= count
#             # this_metric_fold[fold] = this_metric_fold[fold]
#
#             plt.plot(np.arange(len(this_metric_fold[fold])), this_metric_fold[fold], color + line_style,
#                      label=fold + " " + str(lr))
#
#
#             print lr, metric, np.min(this_metric_fold[fold])
#
#     plt.title(metric_full_name + "vs. Epoch")
#     plt.xlabel("# of Epoches")
#     plt.ylabel(metric_full_name)
#
#     plt.grid()
#     plt.legend()
#     plt.show()
#     fig.savefig("../plot/" + sys.argv[0].split(".")[0] + "_" + metric + ".png")
#     plt.cla()
