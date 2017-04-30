import os

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

# FRCN_ROOT = '/home/ahliu/Faster-RCNN_TF/'
FRCN_ROOT = '/mnt/c/Users/Dennis/Dropbox/UC Berkeley/Classes/Spring 2017/CS 294-131/project/Faster-RCNN_TF'
PROJECT_ROOT = os.path.join(FRCN_ROOT, '294-131')
MODEL_FILE = FRCN_ROOT + 'VGGnet_fast_rcnn_iter_70000.ckpt'

IM_GROUP = 'set1'
# IM_ROOT = '/data/efros/ahliu/yt-bb2/'
IM_ROOT = os.path.join(PROJECT_ROOT, 'input', IM_GROUP)
IM_FILES = os.listdir(IM_ROOT)
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'output', IM_GROUP)
