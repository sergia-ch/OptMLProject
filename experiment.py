import argparse
# one experiment!

import waitGPU
import os
import sys

# for environ
import os

# only using device 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

parser = argparse.ArgumentParser(description='Run the experiment')
parser.add_argument('--optimizer', type=str, help='Optimizer')
parser.add_argument('--architecture', type=str, help='Net layers')
parser.add_argument('--image_side', type=int, help='Image side')
parser.add_argument('--giveup', type=int, help='After this number iterations w/o success stop trying')
parser.add_argument('--accuracy_threshold', type=float, help='Accuracy below which nn is discarded and the experiment is repeat')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--repetitions', type=int, help='Number of experiments to run')
parser.add_argument('--train_batch_size', type=int, help='Train batch size')

# optimizer args
#R, p, gamma, ro, learning_rate, beta1, beta2, epsilon
parser.add_argument('--R', type=float)
parser.add_argument('--gamma', type=float)
parser.add_argument('--p', type=float)
parser.add_argument('--ro', type=float)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--beta1', type=float)
parser.add_argument('--beta2', type=float)
parser.add_argument('--epsilon', type=float)

# parsing arguments
args = parser.parse_args()

# FILENAME for output
params_describe = "_".join([x + "-" + str(y) for x, y in vars(args).items()]) + ".output"

# writing something there if it doesn't exist
# if exists, exiting
if os.path.isfile(params_describe):
  if open(params_describe, 'r').read() != 'Nothing[':
    print('Already exists')
    sys.exit(0)

# writing temporary data
open(params_describe, 'w').write('Nothing[')

# waiting for GPU
waitGPU.wait(nproc=1, interval = 10, gpu_ids = [0, 1])

# creating a session...
import tensorflow as tf
tf.reset_default_graph()
# allowing GPU memory growth to allocate only what we need
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
sess = tf.Session(config=config, graph = tf.get_default_graph())

# more imports
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from helpers import *

print('Loading data')

input_shape, output_shape, x_train, x_test, y_train, y_test = get_mnist(7)

print('Creating the model')

# input data: batch w h channels
x = tf.placeholder(tf.float32, shape = (None, *input_shape), name = 'input')

# output labels (vector)
y = tf.placeholder(tf.int64, shape = (None,), name = 'labels')

# one-hot encoded labels
y_one_hot = tf.one_hot(y, output_shape)

# number of layers
Ns = [int(t) for t in args.architecture.split('_')]
print(Ns)

# creating the model
model = FCModelConcat([np.prod(input_shape), *Ns, output_shape], activation = tf.nn.sigmoid)

# model output
output = model.forward(x)

# flatten w * h
l0 = tf.contrib.layers.flatten(x)

# softmax to make probability distribution
logits = tf.nn.softmax(output)

# predicted labels
labels = tf.argmax(logits, axis = 1)

# loss: cross-entropy
loss = tf.losses.softmax_cross_entropy(y_one_hot, logits)

# accuracy of predictions
accuracy = tf.contrib.metrics.accuracy(labels, y)

# list of all parameters
params = tf.trainable_variables()#tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

# get the Hessian of the model
hessian = tf.hessians(loss, model.W)

K = np.sum([np.prod(p.shape).value for p in params])

print('Total parameters:', K)
print('Hessian size: %.2f MB' % (K * K / 1e6))

print('Training')

def get_params_from_args(lst):
  return {x: vars(args)[x] for x in lst}

if args.optimizer == 'adam':
  params = ['learning_rate', 'beta1', 'beta2', 'epsilon']
  gd = tf.train.AdamOptimizer
elif args.optimizer == 'frankwolfe':
  params = ['R', 'p', 'gamma', 'ro']
  gd = StochasticFrankWolfe
elif args.optimizer == 'sgd':
  params = ['learning_rate']
  gd = tf.train.GradientDescentOptimizer
else:
  print('Unknown optimizer ' + args.optimizer)
  sys.exit(0)
train_op = gd(**get_params_from_args(params)).minimize(loss)

# obtaining data...
D = experiment_for_optimizer(train_op, epochs = args.epochs, accuracy_threshold = args.accuracy_threshold, repetitions = args.repetitions,
                         giveup = args.giveup, sess = sess, x_train = x_train, y_train = y_train, x = x, y = y,
                         hessian = hessian, accuracy = accuracy, loss = loss, batch_size = args.train_batch_size,
                        x_test = x_test, y_test = y_test, name = params_describe, p = args.p)

print('Writing results')
f = open(params_describe, "w")
f.write(str(D))
f.close()
