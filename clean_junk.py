import pickle
import os
import numpy as np

data_path = "./processed_dat/"

files = os.listdir(data_path)
f_pairs = list(zip(files[::2], files[1::2]))

for im_f, m_f in f_pairs:

    if os.path.getsize(data_path + m_f) > 0:     
        with open(data_path + m_f, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            m = unpickler.load()

    print(m_f, im_f)

    if np.max(m) == 0.0:
        print('removing')
        os.remove(data_path + m_f)        
        os.remove(data_path + im_f)
    else:
        print("NOT REMOVING THIS!!!")
