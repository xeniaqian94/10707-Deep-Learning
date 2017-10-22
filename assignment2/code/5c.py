import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
import pickle
import os
import re

import time

from train import Model

# python 5aa_plot_single.py ../dump/ 1508612534

dir_file_path = sys.argv[1]  # ../dump/plot_epoch_
files = os.listdir(dir_file_path)

# candidate_model_filename = \
#     [file for file in files if all([tag in file for tag in sys.argv[2].split("_")]) and "model" in file][0]

model = pickle.load(open(dir_file_path + sys.argv[2], "r"))

n_chains = 100
n_visible = model.W.shape[1]

k = 1000
method = "gaussian"

raw_input(n_visible)

fig = plt.figure()

f, axarr = plt.subplots(int(np.sqrt(n_chains)), int(np.sqrt(n_chains)))
plt.gray()

for ind_chain in range(n_chains):
    np.random.seed(int(time.time()))
    v = np.random.uniform(0, 1, n_visible)

    # v = np.random.normal(0, 1, n_visible)
    # v = 1.0 * (v - min(v)) / (max(v) - min(v))

    [h0, v0, h_sample, v_sample, h_prob, v_prob] = model.gibbs_negative_sampling(v, k)

    row = ind_chain / 10
    col = ind_chain % 10
    this_array = np.asarray(v_prob.reshape(np.sqrt(n_visible), np.sqrt(n_visible)))

    axarr[row, col].imshow(this_array)
    axarr[row, col].axis('off')

f.subplots_adjust()
plt.show()
f.savefig("../plot/5c_" + sys.argv[2].split("/")[-1] + "_" + method + ".png")

plt.cla()
plt.clf()
