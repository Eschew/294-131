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

from config import CLASSES, FRCN_ROOT, PROJECT_ROOT, MODEL_FILE, \
    IM_GROUP, IM_ROOT, IM_FILES, OUTPUT_ROOT

os.putenv('CUDA_VISIBLE_DEVICES', '1')

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    net = get_network('VGGnet_test')
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, MODEL_FILE)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

    for im_file in IM_FILES:
        print(im_file)
        im = cv2.imread(os.path.join(IM_ROOT, im_file))

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(sess, net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])
        
        np.save(os.path.join(OUTPUT_ROOT, 'scores', im_file), scores)
        np.save(os.path.join(OUTPUT_ROOT, 'boxes', im_file), boxes)
