from tqdm import tqdm
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import PIL

def fc_layer(x, n, activation = tf.nn.sigmoid):
    """ Fully connected layer for input x and output dim n """
    return tf.contrib.layers.fully_connected(x, n, activation_fn=activation,
    weights_initializer=tf.initializers.lecun_normal(), weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(), biases_regularizer=None, trainable=True)

def CatVariable(shapes, initializer):
    """ List of tensors from a single tensor
    https://github.com/afqueiruga/afqstensorflowexamples/blob/master/afqstensorutils.py
    """
    
    l = np.sum([np.prod(shape) for shape in shapes])
    # V = tf.Variable(tf.zeros(shape=(l,)))
    
    V = tf.Variable(initializer(shape=(l,)))
    
    cuts = []
    l = 0
    for shp in shapes:
        il = 1
        for s in shp: il *= s
        cuts.append(tf.reshape(V[l:(l+il)],shp))
        l += il
    return V, cuts

class FCModelConcat():
    """ Fully-connected network """
    def __init__(self, layer_shapes, activation = tf.nn.relu, initializer = tf.random.truncated_normal):
        """ Initialize with N_neurons (w/o input layer) """
        self.layer_shapes = layer_shapes

        # list of all shapes required
        self.shapes = []
        for i in range(len(layer_shapes) - 1):
            self.shapes.append((layer_shapes[i], layer_shapes[i + 1]))
            self.shapes.append((layer_shapes[i + 1],))
#        print(self.shapes)

        self.activation = activation

        # creating tensors...
        self.W, self.tensors = CatVariable(self.shapes, initializer = initializer)

        # filling weights and biases
        self.biases = []
        self.weights = []

        for i in range(len(self.tensors) // 2):
            self.weights.append(self.tensors[2 * i])
            self.biases.append(self.tensors[2 * i + 1])
        
    def forward(self, l0):
        # layers
        with tf.name_scope('layers'):
            # flattening the input
            z = tf.reshape(l0, (-1, np.prod(l0.shape[1:])))
            i_max = len(self.weights) - 1
            for i in range(i_max + 1):
                z = z @ self.weights[i] + self.biases[i]
                if i < i_max:
                    z = self.activation(z)
            return z

class __FCModel():
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

def get_p_vector_norm(list_of_tensors, order):
    """ p-norm of a list of tensors flattened to a vector """
    # weights as a vector
    weights_flattened = tf.concat([tf.reshape(w, (-1,)) for w in list_of_tensors], axis = 0)

    # p-norm of the weights (must be <= R)
    weight_p_norm = tf.norm(weights_flattened, ord = order)
    return weight_p_norm

def trainable_of(loss):
    # get the Hessian of the model
    return [x for x in tf.trainable_variables() if tf.gradients(loss, [x])[0] is not None]

class StochasticFrankWolfe():
  def __init__(self, R = 0.5, p = 3.1, gamma = 0.5, ro = 0.5):
        # gamma (learning rate)
        self.R = R
        self.p = p
        q =  1. / (1 - 1. / self.p)
        self.q = q
        self.gamma = gamma
        self.ro = ro
  def minimize(self, loss):
        """ Minimize some loss """
        def decrement_weights(weights, gamma, descent_direction):
            """ w = w - how_much """
            # implementing FW update x = (1 - gamma_t) * x + gamma_t * s_t
            frank_wolfe_op = tf.group([w.assign((1. - self.gamma) * w + self.gamma * s)
                           for w, s in zip(weights, descent_direction)])
            return tf.group(frank_wolfe_op)
          
        def LMO(g):
            """ Linear oracle for a list of tensors """

            # overall norm
            g_q_qp_norm = get_p_vector_norm(g, self.q) ** (self.q / self.p)

            g_qp = [tf.abs(g0) ** (self.q / self.p) for g0 in g]
            g_qp_signed = [tf.multiply(g_qp0, tf.sign(g0)) for g_qp0, g0 in zip(g_qp, g)]

            return [-self.R * g_qp_signed0 / g_q_qp_norm for g_qp_signed0 in g_qp_signed]

        # current iteration
        step = tf.Variable(1, dtype = tf.float32, trainable = False)

        # negative gamma -> decreasing
        if self.gamma < 0:
            self.gamma = 2. / (step + 2)

        # negative ro -> decreasing
        if self.ro < 0:
            self.ro = 4. / (step + 8) ** (2. / 3)

        # obtaining weights automatically
        weights = trainable_of(loss)

        # loss gradient
        grads = tf.gradients(loss, weights)

        # temporary variable for grads
        dt = [tf.Variable(x, trainable = False) for x in [tf.zeros_like(g) for g in weights]]

        # p-norm of weights
        weight_p_norm = get_p_vector_norm(weights, self.p)

        # gradients p-norm
        grad_p_norm = get_p_vector_norm(grads, self.p)

         # slack of constraint
        constraint_slack = self.R - weight_p_norm
        
        dt_next = [(1 - self.ro) * dt1 + self.ro * grad1 for dt1, grad1 in zip(dt, grads)]

        descent_direction = LMO(dt_next)
        
        op = decrement_weights(weights, self.gamma, descent_direction)
        #op = decrement_weights(weights, self.gamma, grads)
        with tf.control_dependencies([op]):
            dt_op = tf.group([dt1.assign(dtnext1) for dt1, dtnext1 in zip(dt, dt_next)])

        with tf.control_dependencies([dt_op]):
            step_inc = step.assign(step + 1)
        
        return tf.group([dt_op, op, step_inc])

class OwnGradientDescent():
    def __init__(self, gamma = 0.5, theta = 0.9):
        # gamma (learning rate)
        self.gamma = tf.Variable(gamma, dtype = tf.float32, trainable = False)
        self.theta = theta
        
    def minimize(self, loss):
        """ Minimize some loss """
        def decrement_weights(W, gamma, grads):
            """ w = w - how_much """
            ops = [w.assign(tf.subtract(w, tf.multiply(gamma, grad))) for w, grad in zip(W, grads)]
            return tf.group(ops)
        
        # obtaining weights automatically
        params = trainable_of(loss)
        
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

def shuffle_and_batch(x, batch_size = 100, shuffle = True):
    """ Generate batches of x in random order """
    # shuffling x
    if shuffle: np.random.shuffle(x)
    i = 0
    while i < len(x):
        yield x[i:i+batch_size]
        i += batch_size

assert list(shuffle_and_batch([[1],2,3,4,5,6,7], 2, False)) == [[[1], 2], [3, 4], [5, 6], [7]]

def get_mnist(target_side = 10):
    """ Get mnist dataset downscaled to target_side and normalized """
    # loading mnist
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    output_shape = 10
    input_shape = (28, 28)
    #for i in range(downscale_factor):
        #x_train = x_train[:, ::2, ::2]
        #x_test = x_test[:, ::2, ::2]
        #input_shape = (28, 28)
        #input_shape = (input_shape[0] // 2, input_shape[0] // 2)
    def downscaled(x, w):
        x1 = np.zeros((len(x), w, w))
        for i, x0 in tqdm(enumerate(x)):
            x1[i, :, :] = Image.fromarray(x0).resize((w, w), resample = PIL.Image.BICUBIC)
        return x1
    x_train = downscaled(x_train, target_side)
    x_test = downscaled(x_test, target_side)
    input_shape = (target_side, target_side)
    return input_shape, output_shape, x_train, x_test, y_train, y_test

def epoch(train_op, metrics, batch_size, loss, accuracy, x_train, y_train, sess, x, y, x_test, y_test):
    # do the training
    
    xy = np.arange(len(x_train))
    for batch_xy in shuffle_and_batch(xy, batch_size):
        batch_x, batch_y = x_train[batch_xy], y_train[batch_xy]
        train_loss, train_acc, _ = sess.run([loss, accuracy, train_op], feed_dict = {x: batch_x, y: batch_y})
        
    train_loss, train_acc = sess.run([loss, accuracy], feed_dict = {x: x_train, y: y_train})

    # compute accuracy and loss
    test_loss, test_acc = sess.run([loss, accuracy], feed_dict = {x: x_test, y: y_test})

    # saving data
    metrics['train_loss'] += [train_loss]
    metrics['test_loss'] += [test_loss]
    metrics['train_acc'] += [train_acc]
    metrics['test_acc'] += [test_acc]

def train(operation, epochs_, sess, batch_size, loss, accuracy, x_train, y_train, x, y, x_test, y_test, name):
    """ Train and get metrics """
    # metrics to compute
    metrics = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    
    # initializing weights...
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in tqdm(range(epochs_)):
        epoch(operation, metrics, batch_size, loss, accuracy, x_train, y_train, sess, x, y, x_test, y_test)
    
    #clear_output()
    
    fig, ax1 = plt.subplots()
    ax1.plot(metrics['train_loss'], label = 'train_loss')
    ax1.plot(metrics['test_loss'], label = 'test_loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(metrics['train_acc'], label = 'train_acc')
    ax2.plot(metrics['train_acc'], label = 'train_acc')
    ax2.set_ylabel('accuracy', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    fig.legend()
    plt.savefig('figures/' + name + '.eps', bbox_inches = 'tight')
    plt.show()
    
    return metrics

def get_hessian(hessian, x_train, y_train, x, y, sess):
    """ Get hessian at x, y"""
    H = sess.run(hessian, feed_dict = {x: x_train, y: y_train})
    try:
      eigens = np.real(np.linalg.eig(H)[0][0])
    except:
      eigens = [-1]
    return eigens

def experiment_for_optimizer(gd, epochs, accuracy_threshold, repetitions, giveup, sess, x_train, y_train, x, y, hessian, accuracy, loss, batch_size, x_test, y_test, name, p):
    """ Train repetitions copies of a network with optimizer, output
    Hessian eigenvalues of those satisfying accuracy threshold"""
    
    results = []
    r, i = 0, 0 # found/iterations
    while r < repetitions:
        i += 1
        # giving up if haven't found anything
        if i >= giveup and r == 0:
            return []
        metrics = train(gd, epochs, sess, batch_size, loss, accuracy, x_train, y_train, x, y, x_test, y_test, name)
        train_acc, test_acc = metrics['train_acc'][-1], metrics['test_acc'][-1]
        if train_acc < accuracy_threshold or test_acc < accuracy_threshold:
            print('Error: accuracy is insufficient train=%.2f test=%.2f' % (train_acc, test_acc))
            print('Done: %d/%d/%d' % (r, repetitions, i))
            continue
        assert len(tf.trainable_variables()) == 1, "Must have only one trainable"
        metrics['p_norm'] = sess.run([get_p_vector_norm(tf.trainable_variables(), order = p)])[0]
        eigens = list(get_hessian(hessian, x_train, y_train, x, y, sess))
        r += 1
        print('Done: %d/%d/%d' % (r, repetitions, i))
        metrics['hessian_eigens'] = eigens
        results.append(metrics)
    return results
