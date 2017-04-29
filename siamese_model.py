import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers



# Global constants
IMAGE_SIZE = 256

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
sigmoid_mean = lambda x: tf.sigmoid(x)-tf.reduce_mean(x)

def sigmoid_mean(x):
    y = tf.sigmoid(x)
    return y-tf.reduce_mean(y)

def alexnet_v2_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def alex_feature(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               scope='alexnet_v2'):
  with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
      net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
      net = slim.conv2d(net, 192, [5, 5], scope='conv2')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = slim.conv2d(net, 384, [3, 3], scope='conv3')
      net = slim.conv2d(net, 384, [3, 3], scope='conv4')
      net = slim.conv2d(net, 256, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
      return net

def inference_sim(avec1, avec2, is_training=True, NUM_CLASSES = 2):
    """
    Appearance vectors for first and second batch
    avec1=(batch, 9216)
    avec2=(batch, 9216)
    """
    aggr_vec = tf.concat([avec1, avec2], 1) #produces batch x 18432
    nets = slim.fully_connected(aggr_vec, 10000)
    nets2 = slim.fully_connected(nets, NUM_CLASSES, activation_fn=lambda x:x) #no activation
    return aggr_vec, nets, nets2
    

def featurize(images):
    """Extracts feature vector of (batch, 256, 256, 3)
    to (batch, 9216)"""
    with slim.arg_scope(alexnet_v2_arg_scope()):
        featurization = alex_feature(images)
        return tf.reshape(featurization, (-1, 256*6*6))

def loss(logits, labels):
    with tf.name_scope("Loss"):    
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return loss
    
def train(loss):
    return tf.train.AdamOptimizer(1e-5).minimize(loss)