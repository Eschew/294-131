import os
import scipy
import skimage as sk
import skimage.io as skio
import numpy as np

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
        if len(file_dir) == 0:
            raise Exception("The cateogry %s did not return any images."%category)
        split = int(len(file_dir)*train_percentage)
        # Eliminate cross-contamination
        while True:
            id1 = file_dir[split-1].split('=')[0]
            id2 = file_dir[split].split('=')[0]
            if id1 != id2:
                break
            split += 1
        
        self.training_gallery = file_dir[:split]
        self.testing_gallery = file_dir[split:]
        
    def _batch(self, n, gallery, dim_stack=True, data_aug=False):
        ind = np.arange(len(gallery))
        chosen = np.random.choice(ind, n, replace=False)
        
        images = [] # List of the bounding_box_cropped
        
        for im_file in chosen:
            f = self.dir + gallery[im_file]
            im = skio.imread(f)
            
            tokens = gallery[im_file].split('.jpg')[0].split('=')
            
            xmin, xmax, ymin, ymax = float(tokens[3]), float(tokens[4]), float(tokens[5]), float(tokens[6])
            cropped = im[int(im.shape[0]*ymin):int(im.shape[0]*ymax), int(im.shape[1]*xmin):int(im.shape[1]*xmax)]
            images.append(scipy.misc.imresize(cropped, (IMAGE_SIZE, IMAGE_SIZE)))
        
        
        batch = []
        labels = []
        for i in range(len(images)):
            for j in range(len(images)):
                im1 = images[i]
                im2 = images[j]
                
                if data_aug:
                    print "NOT IMPLEMENTED YET"
                
                if dim_stack:
                    batch.append(np.array([images[i], images[j]]))
                    labels.append(float(i == j))
                
        return self._format_output(batch, labels)
        
        
    def train_batch(self, n, dim_stack=True, data_aug=False):
        """
        n: How many images from gallery to use.
        The actual size of the batch will be the set of 
        every pair of images (n^2).
        
        dim_stack is used for training image on separate networks
            True: images: (n^2, 2, IMAGE_SIZE, IMAGE_SIZE, 3)
                  labels: (n^2, 1)
        
        """
        assert n < 20, "20 images = 400 batch_size"
        return self._batch(n, self.training_gallery, dim_stack=dim_stack, data_aug=data_aug)
    
    def _format_output(self, batch, labels):
        return np.array(batch), np.array([labels]).T
        

    def test_batch(self, n, dim_stack=True, data_aug=False):
        """
        n: How many images from gallery to use.
        The actual size of the batch will be the set of 
        every pair of images (n^2).
        
        dim_stack is used for training image on separate networks
            True: images: (n^2, 2, IMAGE_SIZE, IMAGE_SIZE, 3)
                  labels: (n^2, 1)
        
        """
        assert n < 20, "20 images = 400 batch_size"
        return self._batch(n, self.testing_gallery, dim_stack=dim_stack, data_aug=data_aug)
            
        
            