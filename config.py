import os

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

IMAGE_SIZE = 128
