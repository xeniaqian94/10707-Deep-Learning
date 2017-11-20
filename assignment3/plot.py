from collections import defaultdict

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



# python plot.py [files]
# python plot.py dump/log_-lr_0.5 dump/log_-lr_1.0
#


# python plot.py dump/log_-hidden_size_128_-lr_0.5 dump/log_-hidden_size_128_-lr_1.0 dump/log_-hidden_size_256_-lr_1.0 dump/log_-hidden_size_512_-lr_1.0 dump/log_-lr_1.0_-max_epoch_200


# for each file, it will be like
# epoch 4 start have spent 373.386254072 0.373386254072
# cross entropy for training 6.67976706651
# perplexity for training 796.133644227
# cross entropy for validation 6.33937215418
# perplexity for validation 566.440562378
#


import os
markers=["v","o","^",".",",","<",">"]
colors=['b', 'g', 'r', 'y', 'm', 'c', 'k']
lines=['-',':','.','.-','--']

files=sys.argv[1:]

for (metric, metric_full_name) in [('cross entropy', "Cross entropy"), ('perplexity', "Perplexity")]:
    fig=plt.figure()
    this_metric_fold = dict()

    for marker_ind,file in enumerate(files):
        values=defaultdict(lambda:list())

        for color_ind,fold in enumerate(['training','validation']):
            for line in open(file).readlines():
                if fold in line and metric in line:
                    values[fold]+=[float(line.split()[-1].strip())]


        for color_ind,key in enumerate(values.keys()):  # different folds as different key as different lines
            plt.plot(np.arange(len(values[key])-1), values[key][1:], label=file.split("/")[-1]+"_"+key,markersize=3,markeredgecolor=colors[marker_ind],linestyle=lines[color_ind],c=colors[marker_ind])

    plt.title(metric_full_name + " vs. Epoch")
    plt.xlabel("# of Epoches")
    plt.ylabel(metric_full_name)

    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig("plot/" +"_".join([file.split("/")[-1].strip() for file in files])+"_"+ metric + ".png")
    plt.cla()

