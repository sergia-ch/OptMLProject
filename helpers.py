from tqdm import tqdm
import numpy as np
import tensorflow as tf

def fc_layer(x, n, activation = tf.nn.sigmoid):
    """ Fully connected layer for input x and output dim n """
    return tf.contrib.layers.fully_connected(x, n, activation_fn=activation,
    weights_initializer=tf.initializers.lecun_normal(), weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(), biases_regularizer=None, trainable=True)

class FCModel():
    """ Fully-connected network """
    def __init__(self, N_neurons):
        """ Initialize with N_neurons (w/o input layer) """
        self.N = N_neurons
    def forward(self, l0):
        # layers
        with tf.name_scope('layers'):
            z = tf.reshape(l0, (-1, np.prod(l0.shape[1:])))
            for n in self.N[:-1]:
                z = fc_layer(z, n)
            z = fc_layer(z, self.N[-1], activation = None)
            return z

class OwnGradientDescent():
    def __init__(self, gamma = 0.5, theta = 0.9):
        # gamma (learning rate)
        self.gamma = tf.Variable(gamma, dtype = tf.float32)
        self.theta = theta
        
    def minimize(self, loss, params):
        """ Minimize some loss """
        def decrement_weights(W, gamma, grads):
            """ w = w - how_much """
            ops = [w.assign(tf.subtract(w, tf.multiply(gamma, grad))) for w, grad in zip(W, grads)]
            return tf.group(ops)
        
        # gradients of the loss w.r.t. params
        grads = tf.gradients(loss, params)
        
        # perform gradient descent step
        train_op = decrement_weights(params, self.gamma, grads)
        
        # updating gamma
        upd_op = self.gamma.assign(tf.multiply(self.gamma, self.theta))
        
        return tf.group(train_op, upd_op)

def dz_dw_flatten(z, params):
    """ Calculate dz/dparams and flatten the result """
    return tf.concat([tf.reshape(x, shape = (-1,)) for x in tf.gradients(z, params)], axis = 0)

def iterate_flatten(tensor):
    """ Iterate over flattened items of a tensor """
    if type(tensor) == list:
        for t in tensor:
            for v in iterate_flatten(t):
                yield v
    elif len(tensor.shape) == 0:
        yield tensor
    else:
        for idx in range(tensor.shape[0]):
            for v in iterate_flatten(tensor[idx]):
                yield v

def hessian(loss, params):
    """ Compute a hessian of the loss w.r.t. params """
    with tf.name_scope('hessian'):
        grads = tf.gradients(loss, params)
        grad_components = list(iterate_flatten(grads))
        hessian = [dz_dw_flatten(t, params) for t in tqdm(grad_components)]
        return hessian
