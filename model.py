import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import Model 
import numpy as np
import json

with open('model_config.json', 'rb') as f:
    m_config = json.load(f) 

class MAE(Model):

    def __init__(self):
        super(MAE, self).__init__()

        #Encoder 
        self.encoder = []
        for i, layer in enumerate(m_config['encoder']):
            self.encoder.append(Conv2D(layer['filters'], layer['kernel_size'], padding = layer['padding'], strides = layer['strides'], name = 'e%i'%(i + 1)))
            
        #Decoder
        self.decoder = []
        for i, layer in enumerate(m_config['decoder']):
            self.decoder.append(Conv2DTranspose(layer['filters'], layer['kernel_size'], padding = layer['padding'], strides = layer['strides'], name = 'd%i'%(i + 1)))
        
        #Mask Estimator
        self.masknet = []
        for i, layer in enumerate(m_config['mask']):
            self.masknet.append(Conv2DTranspose(layer['filters'], layer['kernel_size'], padding = layer['padding'], strides = layer['strides'], name = 'm%i'%(i + 1)))
        
        #sigmoid
        self.mo = sigmoid

    def encode(self, inputs):

        #encode 
        x = self.encoder[0](inputs)
        for layer in self.encoder[1:]:
            x = layer(x)
        return x
    
    def decode(self, inputs):

        #decode
        x = self.decoder[0](inputs)
        for layer in self.decoder[1:]:
            x = layer(x)
        return x
    
    def pred_mask(self, inputs, training = False):

        #predict mask
        x = self.masknet[0](inputs)
        for layer in self.masknet[1:]:
            x = layer(x)

        if training:
            return x, self.mo(x)
        return self.mo(x)

    def call(self, inputs, mask = True, training = False):

        x = self.encode(inputs)
        
        if mask:
            return self.pred_mask(x, training = training)            
        return self.decode(x) 

def ae_loss(x, x_pred, mask):

    mask = (mask > 0).astype('int') * 0.9

    return tf.math.reduce_mean(tf.math.square(tf.math.multiply((np.multiply(x, mask) - tf.math.multiply(x_pred, mask)), mask)))

def mask_loss(mask, mask_pred_logits):

    mask = (mask > 0).astype('int') * 0.9

    return tf.math.reduce_mean(binary_crossentropy(mask, mask_pred_logits))