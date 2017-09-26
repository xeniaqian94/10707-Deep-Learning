import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time
import math
import argparse
import matplotlib.pyplot as plt
import pickle
import os

# python 6d_lr.py ../dump/ 1506389922 1506389365 1506388842 1506388334 1506387806

dir_file_path = sys.argv[1]  # ../dump/plot_epoch_


files=os.listdir(dir_file_path)
fig=plt.figure()



for (metric,metric_full_name) in [('ce',"Average Cross Entropy "), ('cr',"Classification Error (in percentage) ")]:
    this_metric_fold = dict()
    for (color,fold) in [("r",'train'), ("g",'valid'), ("b",'test')]:

        for (lr,line_style) in [(0.1,"-."),(0.01,":"),(0.2,"--"),(0.5,"-")]:
            candidate_files=[filename for filename in files if "lr" in filename and fold in filename and metric in filename and str(lr) in filename]
            for ind,filename in enumerate(candidate_files):

                this_list = np.asarray(pickle.load(open(dir_file_path+filename, "r")))
                if metric == "cr":
                    this_list = this_list * 100  # for percentage
                print str(filename), str(type(this_list))
                if not ind:
                    this_metric_fold[fold] = this_list
                else:
                    this_metric_fold[fold] += this_list
                print metric,fold,this_list[:10],this_list[-10:]
            this_metric_fold[fold]/=len(candidate_files)
            this_metric_fold[fold]=this_metric_fold[fold][:30]

            plt.plot(np.arange(len(this_metric_fold[fold])), this_metric_fold[fold], color+line_style,label=fold+" lr "+str(lr))
            this_metric_fold[fold]=None

    plt.title(metric_full_name+"vs. Epoch")
    plt.xlabel("# of Epoches")
    plt.ylabel(metric_full_name)

    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig("../plot/"+sys.argv[0].split(".")[0]+"_"+metric +".png")
    plt.cla()
