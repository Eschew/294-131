import tensorflow as tf
import numpy as np

import os
import utils

ytbb = utils.YTBBQueue('/data/efros/ahliu/yt-bb/', category='car')
im, labels = ytbb.train_batch(5)



