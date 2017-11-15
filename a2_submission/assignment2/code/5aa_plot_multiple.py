import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
import pickle
import os
import re
from train import Model

# python 5aa_plot_multiple.py ../dump/ 1508612534

dir_file_path = sys.argv[1]  # ../dump/plot_epoch_
files = os.listdir(dir_file_path)

fig = plt.figure()
f = None

# as always python plot colors 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'};
#



for (metric, metric_full_name) in [('ce', "Average Cross Entropy ")]:
    # ('cr', "Classification Error")]:


        f, ax = plt.subplots()

        for (color, fold) in [('-', "train_"), (':', "valid_")]:

            # title_post_fix = " at Different Stopping Criteria"
            # for (lr, line_style) in [(150, "y"), (250, "b"), (300, "g"), (350, "c"), (400, "m")]:
            #     # (500, "m"), (600, "y"), (750, "k"),(1000, "r")]:
            #     candidate_file = [filename for filename in files if re.match(".*k_1_max_epoch_" + str(lr) + "$",filename) and fold in filename and metric in filename][0]

            # title_post_fix = " at Different k"
            # for (lr, line_style) in [(1, "r"), (10, "b"), (20, "m")]:
            #     candidate_file = [filename for filename in files if "k_" + str(lr) + "_max_epoch_300" in filename and fold in filename and metric in filename][0]


            # title_post_fix = "with and without pre-training"
            # for (lr, line_style) in [("RBM_1", "r"), ("RBM_2", "b"), ("None", "m"),("AutoEncoder","g"),("Denoise","y")]:
            #     candidate_file = [filename for filename in files if lr in filename and fold in filename and metric in filename][0]

            title_post_fix = "number of hidden units"
            for (lr, line_style) in [("50", "r"), ("100", "b"), ("200", "m"),("500","g")]:
                candidate_file = [filename for filename in files if "n_hidden_"+str(lr)+"_denoise_50_max_epoch_71" in filename and fold in filename and metric in filename][0]




                # if "5" in candidate_file:
                #     candidate_file.replace("300","350")


                this_list = np.asarray(pickle.load(open(dir_file_path + candidate_file, "r")))
                print candidate_file, lr, len(this_list)
                # print this_list
                print

                ax.plot(np.arange(len(this_list)), this_list, color + line_style, label=fold.strip("_") + " " + str(lr),
                        linewidth=1.0)

        ax.set_title(metric_full_name + "vs. Epoch " + title_post_fix)
        ax.set_xlabel("# of Epoches")
        ax.set_ylabel(metric_full_name)

        ax.grid()
        ax.legend()
        plt.show()
        f.subplots_adjust()
        f.savefig("../plot/" + candidate_file.strip("_") + "_" + metric + ".png")

        plt.cla()
        plt.clf()

