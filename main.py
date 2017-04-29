import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network

os.putenv('CUDA_VISIBLE_DEVICES', '1')


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

FRCN_ROOT = '/home/ahliu/Faster-RCNN_TF/'
MODEL_FILE = FRCN_ROOT + 'VGGnet_fast_rcnn_iter_70000.ckpt'

IM_ROOT = '/data/efros/ahliu/yt-bb2/'
IM_FILES = ['ca899NyehXE=14000=person=0.183=0.734=0.051666666=1.jpg']

OUTPUT_ROOT = FRCN_ROOT + '294-131/output/'

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
        np.save(os.path.join(OUTPUT_ROOT, 'boxes', im_file), scores)