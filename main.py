import os, sys
import argparse

import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from networks.factory import get_network

from config import *

os.putenv('CUDA_VISIBLE_DEVICES', '1')

def get_affinity(im1, im2):
  # TODO: replace with siamese network
  return 1

def crop_and_resize(image, bbox, size=(IMAGE_SIZE, IMAGE_SIZE)):
  x1, y1, x2, y2 = bbox.astype(int)
  return cv2.resize(im[x1:x2, y1:y2], size, interpolation=cv2.INTER_AREA)

def compute_track(video):
  track = []
  for im in video:
      scores, bboxes = im_detect(sess, net, im)
      bboxes = np.reshape(boxes.shape[0] * len(CLASSES), 4)
      scores = np.reshape(boxes.shape[0] * len(CLASSES), 1)
      dets = np.concatenate(boxes, scores, axis=1)
      dets = dets[nms(dets, NMS_THRESH), :]
      dets = dets[dets[:, -1] >= CONF_THRESH, :]
      bboxes, scores = dets[:, :4], dets[:, -1]

      if len(track) == 0:
        # TODO: change how first bbox is found
        track.append(bboxes[np.argmax(scores)])
        continue

      track_im = crop_and_resize(im, track[-1])
      bbox_ims = [crop_and_resize(im, bbox) for bbox in bboxes]
      affinities = [get_affinity(track_im, bbox_im) for bbox_im in bbox_ims]
      track.append(bboxes[np.argmax(affinities)])

      # TODO: use kalman filter if affinity too low

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    net = get_network('VGGnet_test')
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, MODEL_FILE)

    videos = [] # TODO: properly load videos
    for video in videos:
      track = process_sequence(video)
