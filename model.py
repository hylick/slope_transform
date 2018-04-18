"""
This file describes the feed forward NN model used for the linear 
regression MVP code
"""

import tensorflow as tf
import numpy as np

class FF():
  
  def init_weights(self, shape):
    weights = tf.random_normal(shape,stddev=1)
    return tf.Variable(weights)
  
  def init_b(self, num_cols):
    return tf.Variable(tf.zeros([num_cols]))
  
  def __init__(self, args, graph):
    self.graph = graph
    self.num_inputs = int(args.num_inputs)
    self.num_neurons = int(args.num_neurons)
    self.num_layers = int(args.num_layers)
    #self.num_outputs = int(args.num_outputs)
    self.num_outputs = self.num_inputs
    self.learning_rate = float(args.learning_rate)
    
    # setup inputs and outputs
    self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None,self.num_inputs]) # shape = [batch_size, num_inputs]
    self.y = tf.placeholder(name='y', dtype=tf.float32, shape=[None,self.num_outputs]) # shape = [batch_size, num_outputs]
 
    # setup weights & biases
    self.weights = []
    self.biases = []
    for i in range(self.num_layers):
      if i == 0:
        self.weights.append(self.init_weights((self.num_inputs,self.num_neurons)))
        self.biases.append(self.init_b(self.num_outputs))
      else: 
        self.weights.append(self.init_weights((self.num_neurons,self.num_neurons)))
        self.biases.append(self.init_b(self.num_neurons))

    # define the graph
    layer_outputs = []
    for i in range(self.num_layers):
      if i == 0:
        layer_outputs.append(tf.matmul(self.x,self.weights[i]) + self.biases[i])
      else:
        layer_outputs.append(tf.matmul(layer_outputs[i-1],self.weights[i]) + self.biases[i])
    self.y_ = layer_outputs[self.num_layers - 1]
    self.y_ = tf.identity(self.y_,name='y_')
    self.loss = tf.losses.mean_squared_error(self.y,self.y_)
    self.loss = tf.identity(self.loss,name='loss')
    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

  def run(self, test=False):
    if test:
      return self.y_, self.loss
    return self.y_, self.loss, self.optimizer
