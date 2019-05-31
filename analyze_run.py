import numpy as np
from matplotlib import pyplot as plt
import os
from create_analyze_runs_helpers import *
import argparse
from config import *

parser = argparse.ArgumentParser(description='Run all experiments in one setting')
parser.add_argument('--setting', type=str, help='Setting to use (small/big/...)')

# parsing arguments
args = parser.parse_args()

### Choosing a setting
setting = args.setting
settings = {'small': small_setting, 'medium': medium_setting, 'big': big_setting, 'batch_size': batch_size_setting}
assert setting in settings, "Please supply a valid setting, one of: " + str(settings.keys())
R = settings[setting]()

# name of the setting, common parameters, parameter groups, all parameters
setting_name, common, param_groups = R
parameters = [x for group in param_groups.values() for x in group]

# group -> what changes
varying = {group: varying_for_optim(param_groups[group]) for group in param_groups.keys()}

print('Which variables are changing for each optimizer?')
print(varying)

# parameter id -> processed file
params_to_processed = []
missing = []
done = 0
for param in parameters:
    fs = process_file(get_file(**param), 'loss_decay_' + setting_name + '_' + shorten_dict(param, filename = True), True)
    print(param)
    print(dict_to_stat(fs))
    print("-----")
    if fs is not None:
        done += 1
    else:
        missing.append(param)
    params_to_processed.append(fs)
    #break
print('Done: %d/%d' % (done, len(parameters)))

# missing files
if missing:
  print("MISSING")
  print(missing)

# all measured quantities
all_metrics = list(params_to_processed[0].keys())

# printing all results (mean over runs/std)
for optim in param_groups.keys():
    data_for_metric = {}
    for m in all_metrics:
#        print("=== OPT %s / METRIC %s ===" % (optim, m))
    
        # labels (parameters used)
        xs = []
        
        # values (arrays) for repetitions
        ys = []
    
        # going over parameters
        for p in param_groups[optim]:
            varying_params = dict_select(p, varying[optim])
            #print(varying_params)
            #print(dict_to_stat(param_to_result(p)))
            res = param_to_result(p, parameters, params_to_processed)
            if res is not None:
                ys.append(res[m])
                xs.append(shorten_dict(varying_params))
        data_for_metric[m] = (xs, ys)
        
    def plot_one(ax, i):
        """ Plot one box plot for metric i at axis ax """
        m = all_metrics[i] # current metric
        xs, ys = data_for_metric[m] # current data
        # plotting
        if xs:
            assert len(xs) == len(ys)
            ax.boxplot(ys, labels = xs)
    subplots(3, 3, all_metrics, plot_one, (5, 5), 'all_' + optim + '_' + setting_name)

# printing best parameters
for optim in param_groups.keys():
    idx = select_best(optim, parameters, param_groups, params_to_processed) # best hyperparameter idx
    p = parameters[idx] # best hyperparameters
    print("BEST FOR", optim, "mean/std")
    print("PARAMS:", dict_select(p, varying[optim]))
    print("RESULTS:", dict_to_stat(params_to_processed[idx]))
    print("")

# final result for all optimizers (best)
data_for_optimizers = {}

# loop over metrics
for m in all_metrics:
    xs = [] # optimizers
    ys = [] # data for optimizers

    # print final results
    for optim in param_groups.keys():
        idx = select_best(optim, parameters, param_groups, params_to_processed) # best hyperparameter
        p = params_to_processed[idx]
        if p is not None:
            xs.append(shorten_name(optim))
            ys.append(p[m])
    
    # saving data...
    data_for_optimizers[m] = (xs, ys)

def plot_one(ax, i):
    """ Plot one box plot for metric i at axis ax """
    m = all_metrics[i] # current metric
    xs, ys = data_for_optimizers[m] # current data
    # plotting
    if xs:
        assert len(xs) == len(ys)
        ax.boxplot(ys, labels = xs)
    if i % 2 == 1:
        ax.yaxis.tick_right()
    if m == 'hessian_eigens_mean':
        #ax.yaxis.set_major_formatter(OOMFormatter(-4, "%1.1f"))
        ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')
        pass
plt.subplots_adjust(wspace = 1)
subplots(4, 2, all_metrics, plot_one, (5, 7), 'best_' + setting_name)
plt.show()
