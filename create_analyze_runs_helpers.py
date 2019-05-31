import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import sys, os

def restore_font():
  """ Restore matplotlib font """
  matplotlib.rc('font', size = 10)

def set_big_font():
  """ Set big font size in matplotlib """
  font = {'family' : 'normal',
      'weight' : 'normal',
      'size'   : 20}

  matplotlib.rc('font', **font)

# folder with .sh scripts and .output files
output_folder = "./output/"
figures_folder = "./output/figures/"

def varying_for_optim(d):
    """ What changes for optimizer? """
    d0 = d[0]
    keys = set()
    for v in d:
        for key, val in v.items():
            if d0[key] != val:
                keys.add(key)
    return list(keys)

def print_nice(y):
    """ Print as a float/other """
    if isinstance(y, float):
        return str(round(y, 10))#'%.2g' % y
    return str(y)

def print_one(**kwargs):
    """ Print run info from kwargs """
    return('python ../experiment.py ' + " ".join(['--' + x + ' ' + print_nice(y) for x, y in kwargs.items()]) + ' &')

def args_in_order_():
    """ Arguments in the order experiment.py expects them """
    # arguments in the correct order
    f = open('experiment.py', 'r').readlines()
    args_in_order = []
    for l in f:
        k = 'parser.add_argument(\'--'
        if l.startswith(k):
            args_in_order.append(l[len(k):].split('\'')[0])
    return args_in_order

def get_file(**kwargs):
    """ Get output filename from kwargs """
    return (output_folder + "_".join([x + '-' + print_nice(kwargs[x] if x in kwargs else None) for x in args_in_order_()])+'.output')

def write_sh_file(setting_name, parameters, common):
    """ Create .sh file with current setting """
    fn = output_folder + 'run_' + setting_name + '.sh'
    out = open(fn, 'w')
    print('OUTPUT: ' + fn)

    def write_to_out(s):
        #print(s)
        out.write(s + '\n')

    it = 0
    write_to_out('#!/bin/bash')
    for params in parameters:
        if it % 6 == 0:
            write_to_out('pids=""')
        write_to_out(print_one(**params))
        #print('echo aba; sleep 3 &')
        write_to_out('pids="$pids $!"')
        write_to_out('sleep 5')

        if it % 6 == 5:
            write_to_out('wait $pids')
            write_to_out('sleep 5')
        it += 1
    it = len(parameters)
    print('Total runs: ', it * common['repetitions'])
    print('Total time (approx): ', common['repetitions'] * 5 * it / 4 / 60)

    out.close()

def arr_of_dicts_to_dict_of_arrays(arr):
    """ Array of dicts to dict of arrays """
    all_keys = arr[0].keys()
    return {key: [v[key] for v in arr] for key in all_keys}

def shorten_name(n):
    """ Shorten aba_caba to a_c """
    return '_'.join([x[:2] if len(x) else '' for x in n.split('_')])

def shorten_dict(d, filename = False):
    """ Shorten dictionary into a string """
    if filename:
        return '_'.join([shorten_name(x) + '-' + str(y) for x, y in d.items()])
    if len(d) == 1:
        return list(d.values())[0]
    return ', '.join([shorten_name(x) + ': ' + str(y) for x, y in d.items()])

def process_dict(d, do_plot, name):
    """ Process one dictionary from  file, return key metrics or plot them """
    d0 = d
    d = arr_of_dicts_to_dict_of_arrays(d)
    all_keys = d.keys()
    metrics = d
    
    results = {key: [] for key in all_keys}
    results['hessian_eigens_mean'] = []
    results['hessian_eigens_Max'] = []
    del results['hessian_eigens']
    
    for i in range(len(d0)):
        for key, val in metrics.items():
            if key == 'hessian_eigens':
                eigens = val[i]
                results['hessian_eigens_mean'].append(np.mean(eigens))
                results['hessian_eigens_Max'].append(np.max(eigens))
            elif isinstance(val[i], list):
                results[key].append(val[i][-1]) # appending LAST loss/accuracy
            else:
                results[key].append(val[i])
    
    if do_plot:
        set_big_font()
        fig, ax1 = plt.subplots(figsize=(5, 2.5))
        ax2 = ax1.twinx()
        
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss', color='b')
        ax1.tick_params('y', colors='b')
        
        ax2.set_ylabel('accuracy', color='r')
        ax2.tick_params('y', colors='r')
        
        for i in range(len(d0)):
            ax1.plot(metrics['train_loss'][i], alpha = 0.3, label = 'train_loss', color = 'b')
            ax1.plot(metrics['test_loss'][i], '--', alpha = 0.3, label = 'test_loss', color = 'b')

            ax2.plot(metrics['train_acc'][i], alpha = 0.3, label = 'train_acc', color = 'r')
            ax2.plot(metrics['test_acc'][i], '--', alpha = 0.3, label = 'test_acc', color = 'r')

        fig.tight_layout()
        #fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
        plt.savefig(figures_folder + name + '.pdf', bbox_inches = 'tight')
        plt.show()
        restore_font()

    return results

def process_file(f, name, do_plot = False):
    """ Process one file """
    if not os.path.isfile(f):
        print('!!! File  missing ' + f)
        return
    
    content = open(f, 'r').read()
    if content.startswith('Nothing['):
        print('!!! File  is empty')
        return
    
    try:
        d = eval(content)
        return process_dict(d, do_plot, name)
    except Exception as e:
        print('!!! File cannot be processed ' + str(e))
        return None
    #return d

def selection_metric(summary):
    """ Summary of one element in params_to_processed[], a number
    Using mean test accuracy over runs
    """
    return np.mean(summary['test_acc']) if summary is not None else -1

def param_to_result(param, parameters, params_to_processed):
    """ Parameter setting to results summary """
    idx = parameters.index(param)
    return params_to_processed[idx]

def select_best(optim, parameters, param_groups, params_to_processed):
    """ Select best parameters for an optimizer """
    
    metrics = [selection_metric(param_to_result(p, parameters, params_to_processed)) for p in param_groups[optim]]
    best_idx = np.argmax(metrics)
    return parameters.index(param_groups[optim][best_idx])

def arr_to_stat(arr):
    """ Array -> mean, std """
    return (np.mean(arr), np.std(arr))

def dict_to_stat(d):
    if d is None:
        return None
    """ Dict key-> arr TO key -> mean, std"""
    return {x: arr_to_stat(y) for x, y in d.items()}

def dict_select(d, ks):
    """ Select only keys from ks from dict d """
    return {x: d[x] for x in ks}

def subplots(n, m, name, fcn, figsize = (10, 13), figname = None):
    """ Plot many subplots Width m, Height n, fcn: thing to plot """
    fig, axs = plt.subplots(n, m, figsize=figsize)
    axs = axs.ravel()
    i = 0
    while i < len(name):
        fcn(axs[i], i)
        axs[i].set_title(shorten_name(name[i]))
        if i == len(name) - 1 or i == len(name) - 2:
            plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=90)
        else:
            axs[i].set_xticks([])
        i += 1
    while i < n * m:
        fig.delaxes(axs[i])
        i += 1
    plt.show()
    if figname is not None:
        fig.savefig(figures_folder + figname + '.pdf', bbox_inches = 'tight')
    return fig
