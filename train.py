import tensorflow as tf 
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta

from model import MAE, ae_loss, mask_loss
from dat_gen import gen_dat

import numpy as np

import argparse
import pickle
import os 

parser = argparse.ArgumentParser()

parser.add_argument("-opt", "--optimizer", help = "optimizer", type = str)
parser.add_argument("-lr", "--l_rate", help = "learning rate", type = float)
parser.add_argument("-ep", "--epochs", help = "number of epochs", type = int)
parser.add_argument("-dp", "--data_path", help = "optimizer", type = str)


args = parser.parse_args()

l_rate = 0.02
epochs = 100
data_path = "./processed_dat/"

if args.l_rate:

    l_rate = args.l_rate

    print("setting learning rate to : %f"%(args.l_rate))


if args.epochs:

    epochs = args.epochs

    print("epochs epochs set to : %i"%(args.epochs))


if args.data_path:

    data_path = args.data_path

    print("dataset path set to : " + args.data_path)

opt = None

if args.optimizer:

    op_name = args.optimizer.lower()

    if op_name == "rmsprop":
        opt = RMSprop(lr)
    elif op_name == "adagrad":
        opt = Adagrad(l_rate)
    elif op_name == "adaelta":
        opt = Adaelta(l_rate)
    else:
        opt = Adam(l_rate)

    print("Using " + args.optimizer + " with learning rate : %f"%(l_rate))

else:

    opt = Adam(l_rate)

    print("Using adam with learning rate : %f"%(l_rate))

files = os.listdir(data_path)
f_pairs = list(zip(files[::2], files[1::2]))

def train(_l_rate = l_rate, _opt = opt, _epochs = epochs, _f_pairs= f_pairs, _data_path = data_path):
    for i in range(_epochs):
        for im_f, m_f in _f_pairs:

            if os.path.getsize(_data_path + im_f) > 0:     
                with open(_data_path + im_f, 'rb') as im_f:
                    unpickler = pickle.Unpickler(im_f)
                    im = unpickler.load()

            if os.path.getsize(_data_path + m_f) > 0:     
                with open(_data_path + m_f, 'rb') as m_f:
                    unpickler = pickle.Unpickler(m_f)
                    m = np.expand_dims(unpickler.load(), axis = -1)

            im_dat = gen_dat(im, overlaps = 1)
            print(im_dat.shape)
            m_dat = gen_dat(m, overlaps = 1)

            del im
            del m 
            #debug
            for e,f in zip(im_dat, m_dat):
                plt.imshow(e)
                plt.show()
                plt.imshow(f)
                plt.show()

train()
        