import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
import pickle
import os
import re
# python 6f_parameters.py ../dump/ 1506389922 1506389365 1506388842 1506388334 1506387806

dir_file_path = sys.argv[1]  # ../dump/plot_epoch_
parameter=sys.argv[2]

files=os.listdir(dir_file_path)
fig=plt.figure()

parameter_name_value=dict()
parameter_name_value['lr']=[0.01,0.02,0.03,0.04,0.05,0.07,0.09,0.1,0.2,0.3,0.4,0.5,0.7]
parameter_name_value['momentum']=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
parameter_name_value['hidden_layer_1_dimension']=[ 20,50,100,200,300,400,500]
parameter_name_value['epoch']=[20,30,50,70,100,125,150,200,300]
parameter_name_value['lbd']=[0.001,0.002,0.003,0.004,0.005,0.007,0.01]


# for lr in 0.01 0.02 0.03 0.04 0.05 0.07 0.09 0.1 0.2 0.3 0.4 0.5 0.7; do nohup condor_run "python train.py -lr $lr $stringg"  & done
#
# for momentum in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do nohup condor_run "python train.py -momentum $momentum $stringg"  & done
#
# for unit in 20 50 100 200 300 400 500; do nohup condor_run "python train.py -hidden_layer_1_dimension $unit $stringg"  & done
#
# for epoch in 20 30 50 70 100 125 150 200 300; do nohup condor_run "python train.py -max_epoch $epoch $stringg" & done
#
# for lbd in 0.001 0.002 0.003 0.004 0.005 0.007 0.01; do nohup condor_run "python train.py -lbd $lbd $stringg" & done

for (metric,metric_full_name) in [('ce',"Average Cross Entropy "), ('cr',"Classification Error (in percentage) ")]:
    for value in parameter_name_value[parameter]:
        filename=[filename for filename in files if re.match(".*"+str(parameter)+"_" + str(value) + "_hidden_layer_2_dimension_100$",filename) and "valid" in filename and metric in filename][0]

        # filename = [filename for filename in files if re.match(".*" + str(parameter) + "_" + str(value) + "$",filename) and "valid" in filename and metric in filename][0]

        this_list=np.asarray(pickle.load(open(dir_file_path + filename, "r")))
        if metric == "cr":
            this_list = this_list * 100  # for percentage

        print filename, metric_full_name, str(np.average(this_list[-3:])), str(len(this_list))
        print
        # print this_list[-10:]
        # print

# for (metric,metric_full_name) in [('ce',"Average Cross Entropy "), ('cr',"Classification Error (in percentage) ")]:
#     this_metric_fold = dict()
#     for (color,fold) in [("r",'train'), ("g",'valid'), ("b",'test')]:
#         # for (lr,line_style) in [(0.1,"-."),(0.01,":"),(0.2,"--"),(0.5,"-")]:
#         #     candidate_files = [filename for filename in files if re.match(".*lr_" + str(lr) + "$",filename) and fold in filename and metric in filename and str(lr) in filename]
#
#         # for (lr,line_style) in [(0.0,"-."),(0.5,"--"),(0.9,"-")]:
#         #     candidate_files=[filename for filename in files if re.match(".*lr_0.01_momentum_" + str(lr) + "$",filename) and fold in filename and metric in filename and str(lr) in filename]
#
#
#         for (lr,line_style) in [(20,"-."),(100,":"),(200,"--"),(500,"-")]:
#             candidate_files = [filename for filename in files if re.match(".*lr_0.01_momentum_0.5_hidden_layer_1_dimension_" + str(lr) + "$",filename) and fold in filename and metric in filename and str(lr) in filename]
#
#             count=0
#             for ind,filename in enumerate(candidate_files):
#
#                 this_list = np.asarray(pickle.load(open(dir_file_path+filename, "r")))
#                 if metric == "cr":
#                     this_list = this_list * 100  # for percentage
#                 print str(filename), str(type(this_list))
#                 if not ind:
#                     this_metric_fold[fold] = this_list
#                 else:
#                     if len(this_list)==len(this_metric_fold[fold]):
#                         count+=1
#                         this_metric_fold[fold] += this_list
#                 print metric,fold,this_list[:10],this_list[-10:]
#             this_metric_fold[fold]/=count
#             this_metric_fold[fold]=this_metric_fold[fold][:30]
#
#             plt.plot(np.arange(len(this_metric_fold[fold])), this_metric_fold[fold], color+line_style,label=fold+" lr "+str(lr))
#             this_metric_fold[fold]=None
#
#     plt.title(metric_full_name+"vs. Epoch")
#     plt.xlabel("# of Epoches")
#     plt.ylabel(metric_full_name)
#
#     plt.grid()
#     plt.legend()
#     plt.show()
#     fig.savefig("../plot/"+sys.argv[0].split(".")[0]+"_"+metric +".png")
#     plt.cla()
