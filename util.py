import os
import scipy
import scipy.misc
import skimage as sk
import skimage.io as skio
import numpy as np
import collections
import glob

from config import *

# GLOBAL CONSTANTS
IMAGE_SIZE = 256


class YTBBQueue():
    def __init__(self, directory, train_percentage=.95, category=""):
        """
        Directory of images, directory should have '/' at end
        Category [list of string]: To only load images of a specific cateogry. 
               If none, then all will be loaded
        """
        assert train_percentage < 1.
        assert train_percentage > 0.
        self.dir = directory
        
        file_dir = [i for i in os.listdir(directory) if '.jpg' in i and i.split("=")[3] in category]
        self.image_sets = collections.defaultdict(list)
        if len(file_dir) == 0:
            raise Exception("The cateogry %s did not return any images."%category)
        
        # Set up tree structure
        for im_file in file_dir:
            tokens = im_file.split('=')
            self.image_sets[tokens[0]+"="+tokens[1]].append(im_file)
            
        split = int(len(self.image_sets.keys())*train_percentage)
        
        self.training_gallery = list(self.image_sets)[:split]
        self.testing_gallery = list(self.image_sets)[split:]
    
    def train_examples(self, batch_size, k, aug=True):
        """
        batch_size: Number of galleries to access
        k: Number of images to pick from gallery
        Every combination of image
        Returns ((batch_size*k)**2, im_size, im_size, 3) for the first set of images
        and ((batch_size*k)**2, im_size, im_size, 3) for the second set of images
        
        There are a total of batch_size*k*k positive examples
        
        """
        num_ids = len(self.training_gallery)
        idx = np.random.choice(np.arange(num_ids), size=batch_size, replace=False)
        ids = [self.training_gallery[i] for i in idx]
        
        selected_ims = []
        
        for i in ids:
            num_ims = len(self.image_sets[i])
            idx2 = np.random.choice(np.arange(num_ims), size=k, replace=True)
            idx2 = [self.image_sets[i][j] for j in idx2]
            for j in idx2:
                im = skio.imread(self.dir+j)
                tokens = j.split(".jpg")[0].split("=")
                xmin = max(int(float(tokens[4])*IMAGE_SIZE)-1, 0)
                xmax = min(int(float(tokens[5])*IMAGE_SIZE)+1, IMAGE_SIZE)
                ymin = max(int(float(tokens[6])*IMAGE_SIZE)-1, 0)
                ymax = min(int(float(tokens[7])*IMAGE_SIZE)+1, IMAGE_SIZE)
                
                im = im[ymin:ymax, xmin:xmax]
                im = scipy.misc.imresize(im, (IMAGE_SIZE, IMAGE_SIZE))
                # im = (im-np.mean(im)) #mean center pixels
                selected_ims.append((i, im))# imageset, image x256 x256
        
        labels = []
        batch1 = []
        batch2 = []
        for i in selected_ims:
            for j in selected_ims:
                if i[0] == j[0]:
                    labels.append([1., 0.])
                else:
                    labels.append([0., 1.])
                
                batch1.append(i[1])
                batch2.append(j[1])
         
        return self.__batch(batch1, batch2, labels)
        
              
                
    def __batch(self, batch1, batch2, labels):
        """ batch1: list of len batch_size with 256,256,3 images
            batch2: list of len batch_size with 256,256,3 images
            labels: list of 1., 0. of len batch_size
            returns: (batch_size, 256, 256, 3), (batch_size, 256, 256, 3), 
                (batch_size, 1) np arrays
            
        """
        assert len(batch1) == len(labels), "Label and batches dim don't line up."
        assert len(batch2) == len(labels), "Label and batches dim don't line up."
        batch1 = np.array(batch1).reshape((-1, 256, 256, 3))
        batch2 = np.array(batch2).reshape((-1, 256, 256, 3))
        labels = np.array(labels).reshape((-1, 2))
        return batch1, batch2, labels
        
    def __len__(self):
        """
        Length of training and testing gallery
        """
        return len(self.training_gallery)+len(self.testing_gallery)

def load_video(yt_id_obj_id):
  ims = glob.glob(os.path.join(DATA_DIR, yt_id_obj_id)+"*")
  ims.sort(key=lambda x: float(x.split("=")[4]))
  frames = []
  for i in ims:
    im = cv2.imread(i)
    frames.append(cv2.imread(i));
  return frames
        