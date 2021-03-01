import tensorflow as tf 
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta

from model import 
import dat_gen

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-opt", "--optimizer", help = "optimizer", type = str)
parser.add_argument("-lr", "--l_rate", help = "learning rate", type = float)


args = parser.parse_args()

l_rate = 0.02

if args.l_rate:

    l_rate = args.l_rate

    print("setting learning rate to : " + args.l_rate)

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


