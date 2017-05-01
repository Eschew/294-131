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

os.putenv('CUDA_VISIBLE_DEVICES', '2')

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
    
    
    
def get_affinity(im1, im2):
  # returns first logit, which is the prediction of same object
  return predict(im1, im2)[0][0]

def crop_and_resize(im, bbox, size=(IMAGE_SIZE, IMAGE_SIZE)):
  x1, y1, x2, y2 = bbox.astype(int)
  return cv2.resize(im[x1:x2, y1:y2], size, interpolation=cv2.INTER_AREA)

def compute_track(video):
  track = []
  track_debug = []
  print len(video)
  for i, im in enumerate(video):
      scores, bboxes = im_detect(sess, net, im)
      bboxes = bboxes.reshape(bboxes.shape[0] * len(CLASSES), 4)
      scores = scores.reshape(scores.shape[0] * len(CLASSES), 1)
      print(i, 'proposals', bboxes.shape[0])
      dets = np.concatenate((bboxes, scores), axis=1)
      dets = dets[nms(dets, NMS_THRESH), :]
      dets = dets[dets[:, -1] >= CONF_THRESH, :]
      bboxes, scores = dets[:, :4], dets[:, -1]
      print(i, 'proposals thresh', bboxes.shape[0])
      np.save(os.path.join(OUTPUT_ROOT, "track%d.npy"%i), bboxes)
      np.save(os.path.join(OUTPUT_ROOT, "track%ds.npy"%i), scores)
      # track_debug.append(np.concatenate(
      #   (bboxes, np.reshape(scores, scores.shape + (1,))), axis=1))
   
      if len(track) == 0:
        # TODO: change how first bbox is found
        i = np.argmax(scores)
        track.append(np.concatenate((bboxes[i], np.array([scores[i]]))))
        continue
      # import pdb; pdb.set_trace()
      track_im = crop_and_resize(im, track[-1][:4])
      bbox_ims = [crop_and_resize(im, bbox) for bbox in bboxes]
      #affinities = [get_affinity(track_im, bbox_im) for bbox_im in bbox_ims]
      #i = np.argmax(affinities)
      #track.append(np.concatenate((bboxes[i], np.array([scores[i]]))))
        
  # TODO: use kalman filter if affinity too low

  return np.array(track), np.array(track_debug)

  
    
def load_videos(yt_id_obj_id):
  ims = glob.glob(os.path.join(DATA_DIR, yt_id_obj_id)+"*")
  ims.sort(key=lambda x: float(x.split("=")[4]))
  frames = []
  for i in ims:
    im = cv2.imread(i)
    frames.append(cv2.imread(i));
  return frames

if __name__ == '__main__':
    
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    net = get_network('VGGnet_test')
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver.restore(sess, MODEL_FILE)
    
    with tf.variable_scope("siamese/"):
        setup()
        feature1 = sm.featurize(pl_inp1)
        feature2 = sm.featurize(pl_inp2)
        smax = tf.nn.softmax(sm.inference_sim(feature1, feature2)[2])
    var = tf.contrib.framework.get_variables_to_restore()
    var = [v for v in var if ("siamese/" in v.name)]
    saver = tf.train.Saver(var)
    # saver.restore(sess, SIAMESE_WEIGHTS)
    
    videos = [load_videos("AA8Besu7Qds=0")]
    tracks = []
    tracks_debug = []
    for video in videos:
      track, track_debug = compute_track(video)
      tracks.append(track)
      tracks_debug.append(track_debug)
    tracks = np.array(tracks)
    tracks_debug = np.array(tracks_debug)
    #import pdb; pdb.set_trace()
    print np.array(tracks).shape
    print np.array(tracks_debug).shape
    np.save(os.path.join(OUTPUT_ROOT, 'tracks'), np.array(tracks))
    np.save(os.path.join(OUTPUT_ROOT, 'tracks_debug'), np.array(tracks_debug))
