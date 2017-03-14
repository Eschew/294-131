import os
import scipy
import skimage as sk
import skimage.io as skio
import numpy as np
import collections

# GLOBAL CONSTANTS
IMAGE_SIZE = 256

class YTBBQueue():
    def __init__(self, directory, train_percentage=.95, category=""):
        """
        Directory of images, directory should have '/' at end
        Category: To only load images of a specific cateogry. If none,
                then all will be loaded
        """
        assert train_percentage < 1.
        assert train_percentage > 0.
        self.dir = directory
        
        file_dir = [i for i in os.listdir(directory) if '.jpg' in i and category in i]
        self.image_sets = collections.defaultdict(list)
        if len(file_dir) == 0:
            raise Exception("The cateogry %s did not return any images."%category)
        
        # Set up tree structure
        for im_file in file_dir:
            tokens = im_file.split('=')
            self.image_sets[tokens[0]].append(im_file)
            
        split = int(len(self.image_sets.keys())*train_percentage)
        
        self.training_gallery = self.image_sets.keys()[:split]
        self.testing_gallery = self.image_sets.keys()[split:]
        
    def _batch(self, n, k, gallery, dim_stack=True, data_aug=False):
        """ Args:
              n - The number of image sets to collect
              k - The number of images to take from an image set
              dim_stack - Whether to stack the dimensions.
              data_aug - Whether to use data augmentation
        """
        ind = np.arange(len(gallery))
        chosen = np.random.choice(ind, n, replace=False)
        
        images = [] # List of the bounding_box_cropped
        
        for ind in chosen:
            image_set_id = gallery[ind]
            images_from_set = self.image_sets[image_set_id]
            chosen2 = np.random.choice(np.arange(len(images_from_set)), k, replace=True)
            files = [images_from_set[ind2] for ind2 in chosen2]
            for fi in files:
                im = skio.imread(self.dir+fi)
                tokens = fi.split('.jpg')[0].split('=')
            
                xmin, xmax, ymin, ymax = float(tokens[3]), float(tokens[4]), float(tokens[5]), float(tokens[6])
                
                xmin = max(int(im.shape[0]*xmin)-1, 0)
                xmax = min(int(im.shape[0]*xmax)+1, im.shape[0])
                ymin = max(int(im.shape[1]*ymin)-1, 0)
                ymax = min(int(im.shape[1]*ymax)+1, im.shape[0])
                cropped = im[ymin:ymax, xmin:xmax]
                
                im = scipy.misc.imresize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
                images.append((im, image_set_id))
        
        batch = []
        labels = []
        for i in range(len(images)):
            for j in range(len(images)):
                im1, im1_id = images[i][0], images[i][1]
                im2, im2_id = images[j][0], images[j][1]
                
                if im1_id == im2_id:
                    labels.append(1.)
                else:
                    labels.append(0.)
                
                if data_aug:
                    print "NOT IMPLEMENTED YET"
                
                if dim_stack:
                    batch.append(np.stack([im1, im2], 0))
                 
                
        return self._format_output(batch, labels)
        
        
    def train_batch(self, n, k=3,dim_stack=True, data_aug=False):
        """
        n: How many image_sets to use.
        k: How many images to take from an image_set
        
        The actual size of the batch will be the set of 
        every pair of images ((nk)^2).
        
        dim_stack is used for training image on separate networks
            True: images: ((nk)^2, 2, IMAGE_SIZE, IMAGE_SIZE, 3)
                  labels: ((nk)^2, 1)
        
        """
        assert n < 20, "20 images = 400 batch_size"
        return self._batch(n, k, self.training_gallery, dim_stack=dim_stack, data_aug=data_aug)
    
    def _format_output(self, batch, labels):
        return np.array(batch), np.array([labels]).T
        

    def test_batch(self, n, k=3, dim_stack=True, data_aug=False):
        """
        n: How many image_sets to use.
        k: How many images to take from an image_set
        
        The actual size of the batch will be the set of 
        every pair of images ((nk)^2).
        
        dim_stack is used for training image on separate networks
            True: images: ((nk)^2, 2, IMAGE_SIZE, IMAGE_SIZE, 3)
                  labels: ((nk)^2, 1)
        
        """
        assert n < 20, "20 images = 400 batch_size"
        return self._batch(n, k, self.testing_gallery, dim_stack=dim_stack, data_aug=data_aug)
            
        
            