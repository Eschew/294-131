import os
import numpy as np

from config_locals import HOME_DIR

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

FRCN_ROOT = os.path.join(HOME_DIR, 'Faster-RCNN_TF')
MODEL_FILE = os.path.join(FRCN_ROOT, 'VGGnet_fast_rcnn_iter_70000.ckpt')

PROJECT_ROOT = os.path.join(HOME_DIR, '294-131')
INPUT_ROOT = os.path.join(PROJECT_ROOT, 'input')
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'output')

SET_NAME = 'set1'
INPUT_SET = os.path.join(INPUT_ROOT, SET_NAME)
IM_FILES = os.listdir(INPUT_SET)
OUTPUT_SET = os.path.join(OUTPUT_ROOT, SET_NAME)
OUTPUT_SET_SCORES = os.path.join(OUTPUT_SET, 'scores')
OUTPUT_SET_BOXES = os.path.join(OUTPUT_SET, 'boxes')
if not os.path.exists(OUTPUT_SET):
	os.mkdir(OUTPUT_SET)
	os.mkdir(OUTPUT_SET_SCORES)
	os.mkdir(OUTPUT_SET_BOXES)

IMAGE_SIZE = 256

NMS_THRESH = 0.3
CONF_THRESH = 0.4
AFF_THRESH = 0.5

SIAMESE_WEIGHTS="/home/ahliu/294-131/checkpoints/siamese-multi-10000"
DATA_DIR = "/data/efros/ahliu/yt-bb4"
VIDEO_NAMES = ["Z3KMX_N6WSg=1"]

# Kalman Filter
P_0 = 256**2/12*np.identity(4) # initial variance = Uniform([0, 255])
Q = 0.001*np.identity(4)       # noise for each state x
R = 1.000*np.identity(4)       # noise for each observation z
