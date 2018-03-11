"""
    1._ Imports
"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from tensorflow.contrib import rnn

"""
Def fully conected layer
"""
def FullyConnected(x, output_dim, scope):
    ##create variables within a scope and specify reuse :
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable("weights", [x.shape[1], output_dim], initializer=tf.random_normal_initializer())
        b = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b

"""
*************  LSTM model  *************
Multilayer LSTM with options (,, use_dropout)
inputs :
    - num_layers  : number of hidden layers in the cells
    - layer_dim   : dimension of the hidden layers
    - use_dropout : boolean if we want to add dropout or not
    - dropout_probs : need to be the same length then num_layers
    specify how much dropout for each layer

outputs :
    - logits :
"""
def Multilayer_LSTM(x, num_layers, layer_dim, output_shape, timesteps, use_dropout, dropout_probs):
    ## Safety check for dropout :
    if use_dropout and not dropout_probs :
        print('Please you need to fill the dropout_probs array with probabilities')
        return
    if use_dropout and dropout_probs and len(dropout_probs)!=num_layers :
        print('You want to use dropout but the dropout_probs needs to have num_layers length')
        return
    ## Unstack the input :
    x = tf.unstack(x, timesteps, 1)
    # Start building the network :
    with tf.variable_scope('lstm_cells', reuse=tf.AUTO_REUSE) as scope:
        lstm_cell = [] # array to fill with the cells consturction
        for i in range(num_layers):
            cell = rnn.BasicLSTMCell(layer_dim[i], forget_bias=1.0)
            if use_dropout :
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_probs[i])
            lstm_cell.append(cell)
        # construct the rnn structure :
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
    # Get the output :
    with tf.variable_scope('rnn_structure', reuse=tf.AUTO_REUSE) as scope:
        lstm_outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    predictions = lstm_outputs[-output_shape:]
    return predictions
