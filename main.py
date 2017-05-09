import os, sys, glob
import pickle as pkl

import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import skimage.io as skio

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from networks.factory import get_network

from config import *
from util import load_video
import siamese_model as sm
from kalman_filter import KalmanFilter

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

def filter_proposals(scores, bboxes, nms_thresh, conf_thresh):
  bboxes_thresh = np.zeros((0, 4))
  scores_thresh = np.zeros((0, 1))
  for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = bboxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        dets = dets[inds, :]
        cls_boxes, cls_scores = dets[:, :4], dets[:, -1]
        bboxes_thresh = np.concatenate((bboxes_thresh, cls_boxes), axis=0)
        scores_thresh = np.concatenate((scores_thresh, cls_scores[:, np.newaxis]), axis=0)
  return bboxes_thresh, scores_thresh

def compute_track(video):
  track = []
  proposals = []

  object_detected = False
  kalman_filter = None

  for t, im in enumerate(video):
      scores, bboxes = im_detect(sess, net, im)
      bboxes, scores = filter_proposals(scores, bboxes,
          nms_thresh=NMS_THRESH, conf_thresh=CONF_THRESH)
      affinities = np.zeros(scores.shape)

      if not object_detected:
        if bboxes.shape[0] == 0:
          track.append(np.zeros(6))
          proposals.append(np.zeros((1, 6)))
          continue
        else:
          object_detected = True

          i = np.argmax(scores)
          xhat_0 = bboxes[i]
          kalman_filter = KalmanFilter(xhat_0)

          track.append(np.concatenate((bboxes[i], scores[i], affinities[i])))
          proposals.append(np.concatenate([bboxes, scores, affinities], axis=1))
          continue

      xhat, P = kalman_filter.update_time()
      score, affinity = np.zeros(1), np.zeros(1)

      if bboxes.shape[0] > 0:
        track_im = crop_and_resize(im, track[-1][:4])
        bbox_ims = [crop_and_resize(im, bbox) for bbox in bboxes]
        affinities = [get_affinity(track_im, bbox_im) for bbox_im in bbox_ims]
        affinities = np.reshape(affinities, scores.shape)

        i, aff = np.argmax(affinities), np.max(affinities)
        if aff > AFF_THRESH:
          xhat, P = kalman_filter.update_measurement(bboxes[i][:4])
          score, affinity = scores[i], affinities[i]

      track.append(np.concatenate((xhat, score, affinity)))
      proposals.append(np.concatenate((bboxes, scores, affinities), axis=1))

  return track, proposals

if __name__ == '__main__':
    os.putenv('CUDA_VISIBLE_DEVICES', '2')
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
    saver.restore(sess, SIAMESE_WEIGHTS)

    videos = [load_video(video_name) for video_name in VIDEO_NAMES]
    tracks = []
    track_proposals = []
    for i, video in enumerate(videos):
      track, proposals = compute_track(video)
      tracks.append(track)
      track_proposals.append(proposals)

    with open(os.path.join(OUTPUT_ROOT, 'tracks'), 'wb') as f:
        pkl.dump(tracks, f)
    with open(os.path.join(OUTPUT_ROOT, 'track_proposals'), 'wb') as f:
        pkl.dump(track_proposals, f)
