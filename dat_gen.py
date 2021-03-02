import numpy as np 
from skimage import util 

def gen_dat(img, size = [512, 512], overlaps = 3, limit = 100):

    depth = img.shape[-1]
    
    step = int(min(img.shape[0] / size[0], img.shape[1] / size[1]) / overlaps)
    wins = util.view_as_windows(img, [size[0], size[1], depth], step = step)

    return np.reshape(wins, [wins.shape[0] * wins.shape[1], wins.shape[3], wins.shape[4], wins.shape[5]])#[:limit]
