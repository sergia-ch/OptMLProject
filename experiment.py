import argparse
# one experiment!

import waitGPU
import os
import sys

parser = argparse.ArgumentParser(description='Run the experiment')
parser.add_argument('--eta', type=float, help='Learning rate')
parser.add_argument('--rho', type=float, help='SFW averating')
parser.add_argument('--mu', type=float, help='Momentum')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--train_batch_size', type=int, help='Train batch size')
parser.add_argument('--iteration', type=int, help='Iteration')

args = parser.parse_args()
params_describe = "_".join([x + "-" + str(y) for x, y in vars(args).items()]) + ".output"
if os.path.isfile(params_describe):
  print('Already exists')
  sys.exit(0)

waitGPU.wait(nproc=6, interval = 10, gpu_ids = [0])

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from experiment_helpers import *

from dfw.dfw import DFW
from dfw.dfw.losses import MultiClassHingeLoss
from dfw.experiments.utils import accuracy


print('Loading data')

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=args.train_batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=10000, shuffle=True)

print('Creating the model')
model = SimpleCNN().cuda()
svm = MultiClassHingeLoss()
optimizer = DFW(model.parameters(), eta=args.eta, rho_ = args.rho, momentum = args.mu)
print('Training')

D = train(optimizer, model, train_loader, epochs = args.epochs, loss = svm)
D.update(metrics_post_all(train_loader, test_loader = test_loader, model =  model, loss = svm))
#print(D)

print('Writing results')
f = open(params_describe, "w")
f.write(str(D))
f.close()
