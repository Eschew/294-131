import tensorflow as tf
import numpy as np

import os
import utils

# Set up environment variables
os.putenv('CUDA_VISIBLE_DEVICES', '2')

# Global constants
IMAGE_SIZE = 256
DATA_DIR = '/data/efros/ahliu/yt-bb/'

# Network Specific


ytbb = utils.YTBBQueue('/data/efros/ahliu/yt-bb/')
print len(ytbb)

def train():
    

def setup():
    global batch_queue
    batch_queue = utils.YTBBQueue(DATA_DIR)
    

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
