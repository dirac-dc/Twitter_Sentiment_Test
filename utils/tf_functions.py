import numpy as np
from numpy import random
import nltk
#nltk.download_shell()
from nltk.tokenize import word_tokenize
import string
from unidecode import unidecode
#from lib.ops import *

import tensorflow as tf
from data_reader import *

def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    """ Straight-forward convvolutional layer """
    # w is the kernel, b the bias, no strides and VALID padding

    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b

def tdnn(input_, kernels, kernel_features, scope='TDNN'):
    ''' Time Delay Neural Network
    :input:           input float tensor of shape 
                      [(batch_size*num_unroll_steps) x max_word_length x embed_size]
    :kernels:         array of kernel sizes
    :kernel_features: array of kernel feature sizes (parallel to kernels)
    '''
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    # input_ is a np.array of shape ('b', 'sentence_length', 'max_word_length', 'embed_size') we
    # need to convert it to shape ('b * sentence_length', 1, 'max_word_length', 'embed_size') to
    # use conv2D
    # It might not seem obvious why we need to use this small hack at first sight, the reason
    # is that sentence_length will change across the different minibatches, but if we kept it
    # as is sentence_length would act as the number of channels in the convnet which NEEDS to
    # stay the same
    rd = read_data(create=False)
    input_ = tf.reshape(input_, [-1, rd.max_word_length, rd.ALPHABET_SIZE])
    input_ = tf.expand_dims(input_, 1)

    layers = []
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            reduced_length = rd.max_word_length - kernel_size + 1

            # [batch_size * sentence_length x max_word_length x embed_size x kernel_feature_size]
            conv = conv2d(input_, kernel_feature_size, 1,
                          kernel_size, name="kernel_%d" % kernel_size)

            # [batch_size * sentence_length x 1 x 1 x kernel_feature_size]
            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')

            layers.append(tf.squeeze(pool, [1, 2]))

        if len(kernels) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]

    return output

def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        scope: VariableScope for the created subgraph; defaults to "Linear".
        
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
        
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size],
                                 dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output

def softmax(input_, out_dim, scope=None):
    """ SoftMax Output """

    with tf.variable_scope(scope or 'softmax'):
        W = tf.get_variable('W', [input_.get_shape()[1], out_dim])
        b = tf.get_variable('b', [out_dim])

    return tf.nn.softmax(tf.matmul(input_, W) + b)