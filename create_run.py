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

### Creating the `.sh` file
write_sh_file(setting_name, parameters, common)
