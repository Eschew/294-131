import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import siamese_model as sm

import os
import util

# Set up environment variables
os.putenv('CUDA_VISIBLE_DEVICES', '0')

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
def predict(im1, im2):
    with tf.variable_scope("siamese/"):
        feature1 = sm.featurize(pl_inp1)
        feature2 = sm.featurize(pl_inp2)
        (agg, h, logits) = sm.inference_sim(feature1, feature2)
        smax = tf.nn.softmax(logits)

        restorer = tf.train.Saver()
        with tf.Session() as sess:
            restorer.restore(sess, "/home/ahliu/294-131/checkpoints/siamese-multi-1000")
            smax_values = sess.run(smax, feed_dict={pl_inp1:im1, pl_inp2:im2})

    return smax_values
    

def setup():
    global batch_queue
    global pl_inp1
    global pl_inp2
    global pl_exp
    cat = "airplane,bird,boat,bus,car,cat,cow,dog,horse,person,train"
    cats = cat.split(",")
    batch_queue = util.YTBBQueue(DATA_DIR)
    pl_inp1 = tf.placeholder(tf.float32, (None, 256, 256, 3))
    pl_inp2 = tf.placeholder(tf.float32, (None, 256, 256, 3))
    
    pl_exp = tf.placeholder(tf.float32, (None, NUM_CLASSES))
    
    
    
    
    

def main(argv=None):
    setup()
    corr = 0
    incorr = 0
    for _ in range(40):
        (im1, im2, label) = batch_queue.train_examples(3, 4)
        pred = predict(im1, im2)
        n1 = np.argmax(pred, axis=1)
        n2 = np.argmax(label, axis=1)
        for i in range(n1.shape[0]):
            if n1[i] == n2[i]:
                corr += 1
            else:
                incorr += 1
    print corr/(corr+incorr+0.), incorr/(corr+incorr+0.)

if __name__ == '__main__':
    tf.app.run()
