import pandas as pd
import numpy as np 
import cv2

import json
import pickle
import os
import tifffile as tiff

from skimage import io

#paths 
ds_path = './hubmap-kidney-segmentation/train/'
out_path = './processed_dat/'

#get file list 
img_list = [f for f in os.listdir(ds_path) if '.tiff' in f]

#number of splits per axis 
splits = 10

#process individual images
for im_name in img_list:
    
    print(im_name)

    #Save image chunks
    
    x = tiff.imread(ds_path + im_name)
    
    width = np.shape(x)[1]
    height = np.shape(x)[0]
    
    for i, h_split in enumerate(np.split(x[:int(height/splits)*splits,:int(width/splits)*splits], indices_or_sections = splits, axis = 0)):
        for j, hv_split in enumerate(np.split(h_split, indices_or_sections = splits, axis = 1)):
            
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            with open(out_path + '%s_%i_%i.pickle'%(im_name.replace('.tiff', ''), i, j), 'wb') as f:
                h = pickle.dump(hv_split, f)  

            del h
            del hv_split
        del h_split
    del x
    
    shape = (height, width)
    
    #Save mask chunks
    
    with open(ds_path + im_name.replace('.tiff', '.json')) as m_file:
        m_dict = json.load(m_file)
    
    polys = []
    
    for ob in m_dict:
        geom = np.array(ob['geometry']['coordinates'])
        polys.append(geom)
        
    mask = np.zeros(shape)
    
    for i, poly in enumerate(polys):
        cv2.fillPoly(mask, poly, i+1)
        
    print('filled in mask, procesing...')
        
    for i, h_split in enumerate(np.split(mask[:int(height/splits)*splits,:int(width/splits)*splits], indices_or_sections = splits, axis = 0)):
        for j, hv_split in enumerate(np.split(h_split, indices_or_sections = splits, axis = 1)):
            
            with open('./processed_dat/%s_%i_%i_mask.pickle'%(im_name.replace('.tiff', ''), i, j), 'wb') as f:
                h = pickle.dump(hv_split, f)

            del h
            del hv_split
        del h_split
                
    del mask 
    