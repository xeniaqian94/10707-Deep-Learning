import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time
import math
import argparse
import matplotlib.pyplot as plt
import pickle
import re

# python 6ab_load_ce_cr_avg_by_num_samply.py ../dump/
#
# plot_epoch_ 1506389922 1506389365 1506388842 1506388334 1506387806

dir_file_path = sys.argv[1]  # ../dump/plot_epoch_

import os

files = os.listdir(dir_file_path)

for (metric, metric_full_name) in [('ce', "Average Cross Entropy "), ('cr', "Classification Error (in percentage) ")]:
    fig=plt.figure()
    this_metric_fold = dict()
    for fold in ['train', 'valid', 'test']:
        candidate_files = [filename for filename in files if
                           re.match(".*lr_0.02$", filename) and fold in filename and metric in filename]

        print candidate_files
        # for ind, version in enumerate(sys.argv[2:]):
        for ind, filename in enumerate(candidate_files):
            print filename
            this_list = np.asarray(pickle.load(open(dir_file_path + filename, "r")))
            if metric == "cr":
                this_list = this_list * 100  # for percentage
            print str(dir_file_path + filename), str(type(this_list))
            if not ind:
                this_metric_fold[fold] = this_list
            else:
                this_metric_fold[fold] += this_list
            print metric, fold, this_list[:10], this_list[-10:]

        this_metric_fold[fold] /= len(candidate_files)

        plt.plot(np.arange(len(this_metric_fold[fold])), this_metric_fold[fold], label=fold)

    plt.title(metric_full_name + "vs. Epoch")
    plt.xlabel("# of Epoches")
    plt.ylabel(metric_full_name)

    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig("../plot/" + metric + ".png")
    plt.cla()
