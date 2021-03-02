import tensorflow as tf 
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta

from model import MAE, ae_loss, mask_loss
from dat_gen import gen_dat

import argparse
import os 

parser = argparse.ArgumentParser()

parser.add_argument("-opt", "--optimizer", help = "optimizer", type = str)
parser.add_argument("-lr", "--l_rate", help = "learning rate", type = float)
parser.add_argument("-ep", "--epochs", help = "number of epochs", type = int)
parser.add_argument("-dp", "--dat_path", help = "optimizer", type = str)


args = parser.parse_args()

l_rate = 0.02
epochs = 100
dat_path = "./processed_dat/"

if args.l_rate:

    l_rate = args.l_rate

    print("setting learning rate to : " + args.l_rate)


if args.epochs:

    epochs = args.epochs

    print("epochs epochs set to : " + args.epochs)


if args.dat_path:

    dat_path = args.dat_path

    print("dataset path set to : " + args.dat_path)

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

    print("Using " + args.optimizer + " with learning rate : " + l_rate)

else:

    opt = Adam(l_rate)

    print("Using adam with learning rate : " + l_rate)

files = os.list(dat_path)
f_pairs = list(zip(files[::2], files[1::2]))

def train(_l_rate = l_rate, _opt = opt, _epochs = epochs, _f_pairs= f_pairs):
    for i in range(_epochs):
        for im_f, m_f in _f_pairs:
            #needs change
            if os.path.getsize(data_path + f_pairs[n][0]) > 0:     
                with open(data_path + f_pairs[n][0], 'rb') as im_f:
                    unpickler = pickle.Unpickler(im_f)
                    im = unpickler.load()
            #needs change
            if os.path.getsize(data_path + f_pairs[n][1]) > 0:     
                with open(data_path + f_pairs[n][1], 'rb') as m_f:
                    unpickler = pickle.Unpickler(m_f)
                    m = unpickler.load()    

            im_dat = gen_dat(im)
            m_dat = gen_dat(m)

            del im
            del m 
            #debug
            for e,f in zip(im_dat, m_dat):
                plt.imshow(e)
                plt.show()
                plt.imshow(f)
                plt.show()
        