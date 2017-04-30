import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import siamese_model as sm

import os
import util

# Set up environment variables
os.putenv('CUDA_VISIBLE_DEVICES', '0,2')

# Global Constants
IMAGE_SIZE = 256
DATA_DIR = '/data/efros/ahliu/yt-bb4/'
NUM_CLASSES = 2
checkpoint_dir = '/home/ahliu/294-131/checkpoints/siamese-multi'

# Global  Variables
batch_queue = None
pl_inp1 = None
pl_inp2 = None
pl_exp = None

# Network Specific
def train():
    feature1 = sm.featurize(pl_inp1)
    feature2 = sm.featurize(pl_inp2)
    
    (agg, h, logits) = sm.inference_sim(feature1, feature2)
    loss = sm.loss(logits, pl_exp)
    
    train_op = sm.train(loss)
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    
    saver = tf.train.Saver(max_to_keep=20)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(20001):
            (batch1, batch2, labels) = batch_queue.train_examples(3, 4)
            if step%1000 == 0:
                saver.save(sess, checkpoint_dir, global_step=step)
                print "Saved model to: "+checkpoint_dir
            comp_loss = sess.run([loss, train_op], feed_dict={pl_inp1:batch1, pl_inp2:batch2, pl_exp:labels})
            print step, comp_loss
    

def setup():
    global batch_queue
    global pl_inp1
    global pl_inp2
    global pl_exp
    cat = "airplane,bird,boat,bus,car,cat,cow,dog,horse,person,train"
    cats = cat.split(",")
    batch_queue = utils.YTBBQueue(DATA_DIR, category=cats)
    pl_inp1 = tf.placeholder(tf.float32, (None, 256, 256, 3))
    pl_inp2 = tf.placeholder(tf.float32, (None, 256, 256, 3))
    
    pl_exp = tf.placeholder(tf.float32, (None, NUM_CLASSES))
    
    
    
    
    

def main(argv=None):
    setup()
    train()

if __name__ == '__main__':
    tf.app.run()
