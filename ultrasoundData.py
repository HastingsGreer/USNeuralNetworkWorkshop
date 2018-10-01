
import numpy as np
import pickle
import scipy.interpolate
import scipy.misc

from scipy.ndimage.filters import gaussian_filter1d


class UltrasoundData:
    def __init__(self, filename):
        with open(filename, "rb") as file:
            p = pickle.load(file)
            #print(p[0])
            self.rawImages = p[0]
            
            self.rot, self.trans, self.otime = p[1]
            self.otime = np.array(self.otime, dtype=np.float)
            self.otime /= 1000
        
        
        self.rotAngles = np.arctan2(self.rot[:, 2, 0], self.rot[:, 2, 2])
        
        
    def interpToImageTime(self, array_in_tracker_time):
        interpolator = scipy.interpolate.interp1d(self.otime, array_in_tracker_time, assume_sorted=True)
        return interpolator(self.rawImages[0]) #rawImages[0] is the timestamp of each ultrasound capture
    
    def makeData(self):
        self.monoImages = np.sum(self.rawImages[1], 3).astype(np.float)
        
        
        self.angles, self.horizontal, self.vertical, self.length = map(self.interpToImageTime, 
                                                          [self.rotAngles, self.trans[:, 0], self.trans[:, 1], self.trans[:, 2]])
        self.rawangles = self.angles.copy()
        #self.angles = gaussian_filter1d(self.angles, 6)
        
        self.angles = np.concatenate([self.angles, -self.angles])
        self.horizontal = np.concatenate([self.horizontal, -self.horizontal])
        self.vertical = np.concatenate([self.vertical, self.vertical])
        self.length = np.concatenate([self.length, self.length])



        self.data = np.array([scipy.misc.imresize(arr, (100, 100)) for arr in self.monoImages]).reshape(-1, 100, 100, 1) / 255
        del self.monoImages
        self.data = np.concatenate([self.data, np.flip(self.data, 2)])
        
        #uncomment this to experiment with learning position as well as angle
        #self.classes = np.stack([self.angles, self.horizontal / 150, self.vertical/ 150, self.length / 800 - 1]).transpose() 
        

        self.classes = np.array([self.angles]).transpose()

        """
        uncomment this to experiment with learning angles as a one-hot encoded category instead of as a scalar
        self.classes /= 2
        
        self.classes += .5
        self.classes *= 15
        
        self.classes = keras.utils.to_categorical(self.classes.astype(np.int), 30)
        
        """

def stagger(data, classes, n):
    """
    Function to create short sequences from a long time series
    """
    stagger_data = [
        data[i:i + 2 * n + 1]
        for i in range(0, len(data) - 2 * n)
    ]
    
    stagger_classes = classes[n:-n]
    
    stagger_data = np.array([np.concatenate(n) for n in np.array(stagger_data)])
    
    
    return stagger_data, stagger_classes

x = 9