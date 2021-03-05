import numpy as np 

def gen_dat(img, size = [512, 512], batch_size = 100, offsets = None):

    max_offset = img.shape[:-1] - np.array(size) - 1
    
    if offsets is None:
        offsets = np.random.randint(max_offset, size = [batch_size, 2])

    sample = np.array([img[of[0] : of[0] + size[0], of[1] : of[1] + size[1], :] for of in offsets])

    return sample, offsets