import tensorflow as tf 
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta

from model import MAE, ae_loss, mask_loss
from dat_gen import gen_dat

import numpy as np
from progress.bar import Bar
from matplotlib import pyplot as plt 

import argparse
import pickle
import json
import os 

parser = argparse.ArgumentParser()

parser.add_argument("-opt", "--optimizer", help = "optimizer", type = str)
parser.add_argument("-lr", "--l_rate", help = "learning rate", type = float)
parser.add_argument("-ep", "--epochs", help = "number of epochs", type = int)
parser.add_argument("-dp", "--data_path", help = "optimizer", type = str)


args = parser.parse_args()

l_rate = 0.02
epochs = 100
batch_size = 100
data_path = "./processed_dat/"

weight_path = "./.weights/"

if not os.path.exists(weight_path):
    os.mkdir(weight_path)

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

model = MAE()


def train(_l_rate = l_rate, _opt = opt, _epochs = epochs, _f_pairs= f_pairs, _data_path = data_path):

    losses = []
    
    t_vars = None
    print(t_vars)


    for i in range(_epochs):

        print("Epoch %i"%(i))
        
        e_losss = []

        for im_f, m_f in _f_pairs:

            if os.path.getsize(_data_path + im_f) > 0:     
                with open(_data_path + im_f, 'rb') as im_f:
                    unpickler = pickle.Unpickler(im_f)
                    im = unpickler.load()

            if os.path.getsize(_data_path + m_f) > 0:     
                with open(_data_path + m_f, 'rb') as m_f:
                    unpickler = pickle.Unpickler(m_f)
                    m = np.expand_dims(unpickler.load(), axis = -1)

            im_dat, offsets = gen_dat(im, batch_size = batch_size)

            print(im_dat.shape)

            m_dat, offsets = gen_dat(m, batch_size = batch_size, offsets = offsets)

            del im
            del m 
            
            #compute loss
            with tf.GradientTape(persistent = True) as t:
                #For now, we're just training the autoencoder
                #encode 
                im_enc = model.encode(im_dat)
                #decode
                im_pred = model.decode(im_enc)

                if t_vars == None:                
                    t_vars = model.trainable_variables
                    e_vars = [var for var in t_vars if 'e' in var.name]
                    d_vars = [var for var in t_vars if 'd' in var.name]
                    m_vars = [var for var in t_vars if 'm' in var.name]

                #ae loss 
                l_1 = ae_loss(im_dat, im_pred, m_dat)


            enc_grads = t.gradient(l_1, e_vars)
            dec_grads = t.gradient(l_1, d_vars)
            
            print(np.sum(np.array([np.sum(grad.numpy()) for grad in enc_grads])))
            print(np.sum(np.array([np.sum(grad.numpy()) for grad in dec_grads])))

            #apply encoder grads 
            opt.apply_gradients(zip(enc_grads, e_vars))
            #applt decoder grads
            opt.apply_gradients(zip(dec_grads, d_vars))
            
            print(l_1.numpy())
            e_losss.append(l_1)

        #losses.append(l_1)

        model.save_weights(weight_path + "ep_%i"%(i))
        losses.append(np.sum(e_losss))

    print(losses)


train()
        