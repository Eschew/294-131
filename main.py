import os, sys, glob
import argparse

import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import skimage.io as skio

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import siamese_model as sm
import siamese_test

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from networks.factory import get_network

from config import *

os.putenv('CUDA_VISIBLE_DEVICES', '1')

# Global  Variables
pl_inp1 = None
pl_inp2 = None

# Network Specific
def predict(im1, im2):    
    # Inputs two numpy images that are 256x256
    im1 = im1.reshape((1, 256, 256, 3))
    im2 = im2.reshape((1, 256, 256, 3))
    smax_values = sess.run(smax, feed_dict={pl_inp1:im1, pl_inp2:im2})
    
    return smax_values
    

def setup():
    global pl_inp1
    global pl_inp2
    
    pl_inp1 = tf.placeholder(tf.float32, (None, 256, 256, 3))
    pl_inp2 = tf.placeholder(tf.float32, (None, 256, 256, 3))
    pl_exp = tf.placeholder(tf.float32, (None, NUM_CLASSES))

    
    feature1 = sm.featurize(pl_inp1)
    feature2 = sm.featurize(pl_inp2)
    (agg, h, logits) = sm.inference_sim(feature1, feature2)
    smax = tf.nn.softmax(logits)
    
    restorer = tf.train.Saver()
    restorer.restore(sess, "/home/ahliu/294-131/checkpoints/siamese-multi-1000")
    return smax
    
    
    
    
def get_affinity(sess, im1, im2):
  return predict(sess, im1, im2)

def crop_and_resize(im, bbox, size=(IMAGE_SIZE, IMAGE_SIZE)):
  x1, y1, x2, y2 = bbox.astype(int)
  return cv2.resize(im[x1:x2, y1:y2], size, interpolation=cv2.INTER_AREA)

def compute_track(sess, video):
  track = []
  for im in video:
      scores, bboxes = im_detect(sess, net, im)
      bboxes = bboxes.reshape(bboxes.shape[0] * len(CLASSES), 4)
      scores = scores.reshape(scores.shape[0] * len(CLASSES), 1)
      dets = np.concatenate((bboxes, scores), axis=1)
      dets = dets[nms(dets, NMS_THRESH), :]
      dets = dets[dets[:, -1] >= CONF_THRESH, :]
      bboxes, scores = dets[:, :4], dets[:, -1]

      if len(track) == 0:
        # TODO: change how first bbox is found
        track.append(bboxes[np.argmax(scores)])
        continue

      track_im = crop_and_resize(im, track[-1])
      bbox_ims = [crop_and_resize(im, bbox) for bbox in bboxes]
      affinities = [get_affinity(sess, track_im, bbox_im) for bbox_im in bbox_ims]
      track.append(bboxes[np.argmax(affinities)])

      # TODO: use kalman filter if affinity too low
    
def load_videos(yt_id_obj_id):
  ims = glob.glob(os.path.join(DATA_DIR, yt_id_obj_id)+"*")
  ims.sort(key=lambda x: float(x.split("=")[4]))
  frames = []
  for i in ims:
    im = cv2.imread(i)
    frames.append(cv2.imread(i));
  return frames

if __name__ == '__main__':
    # Setup for siamese weights
    setup()
    
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    net = get_network('VGGnet_test')
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, MODEL_FILE)

    videos = [load_videos("ZFSspVdQ_1M=0")]
    tracks = []
    for video in videos:
      track = compute_track(sess, video)
      tracks.append(track)
    np.save(os.path.join(OUTPUT_ROOT, 'tracks'), tracks)
