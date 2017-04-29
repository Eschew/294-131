import os, sys

import matplotlib.pyplot as plt
import numpy as np
import cv2

import _init_paths
from fast_rcnn.nms_wrapper import nms


os.putenv('CUDA_VISIBLE_DEVICES', '1')


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

FRCN_ROOT = '/home/ahliu/Faster-RCNN_TF/'
FRCN_ROOT = '/mnt/c/Users/Dennis/Dropbox/UC Berkeley/Classes/Spring 2017/CS 294-131/project/Faster-RCNN_TF'
PROJECT_ROOT = os.path.join(FRCN_ROOT, '294-131')
MODEL_FILE = FRCN_ROOT + 'VGGnet_fast_rcnn_iter_70000.ckpt'

IM_ROOT = '/data/efros/ahliu/yt-bb2/'
IM_ROOT = os.path.join(PROJECT_ROOT, 'input')
IM_FILES = ['ca899NyehXE=14000=person=0.183=0.734=0.051666666=1.jpg']

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'output')

def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

if __name__ == '__main__':       
    for im_file in IM_FILES:
        print(im_file)
        im = cv2.imread(os.path.join(IM_ROOT, im_file))

        scores = np.load(os.path.join(OUTPUT_ROOT, 'scores', im_file + '.npy'))
        boxes = np.load(os.path.join(OUTPUT_ROOT, 'boxes', im_file + '.npy'))
        
        # Visualize detections for each class
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)
 
    plt.show()

