import tensorflow as tf
import numpy as np

import os
import utils

# Set up environment variables
os.putenv('CUDA_VISIBLE_DEVICES', '2')

# Global constants
IMAGE_SIZE = 256
DATA_DIR = '/data/efros/ahliu/yt-bb/'

ytbb = utils.YTBBQueue('/data/efros/ahliu/yt-bb/', category='car')

im, labels = ytbb.train_batch(5, 2)

    
    
    
