import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import Model 

class MAE(Model):

    def __init__(self, filters = [8, 8, 16, 16, 32]):
        super(MAE, self).__init__()

        #Encoder 
        #512    -> 256 
        self.e1 = Conv2D(filters[0], kernel_size = [3, 3], padding = 'same', strides = [2, 2])
        #256    -> 128 
        self.e2 = Conv2D(filters[1], kernel_size = [3, 3], padding = 'same', strides = [2, 2])
        #128    -> 64 
        self.e3 = Conv2D(filters[2], kernel_size = [3, 3], padding = 'same', strides = [2, 2])
        #64     -> 32 
        self.e4 = Conv2D(filters[3], kernel_size = [3, 3], padding = 'same', strides = [2, 2])
        #32     -> 16
        self.e5 = Conv2D(filters[4], kernel_size = [3, 3], padding = 'same', strides = [2, 2])

        #Decoder
        #16     -> 32
        self.d1 = Conv2DTranspose(filters[3], kernel_size = [3, 3], padding = 'same', strides = [2, 2])
        #32     -> 64
        self.d2 = Conv2DTranspose(filters[2], kernel_size = [3, 3], padding = 'same', strides = [2, 2])
        #64     -> 128
        self.d3 = Conv2DTranspose(filters[1], kernel_size = [3, 3], padding = 'same', strides = [2, 2])
        #128    -> 256
        self.d4 = Conv2DTranspose(filters[0], kernel_size = [3, 3], padding = 'same', strides = [2, 2])
        #256    -> 512
        self.d5 = Conv2DTranspose(3, kernel_size = [3, 3], padding = 'same', strides = [2, 2])

        #Mask Estimator
        #16     -> 64
        self.m1 = Conv2DTranspose(filters[3], kernel_size = [3, 3], padding = 'same', strides = [4, 4])
        #64     -> 256
        self.m2 = Conv2DTranspose(filters[1], kernel_size = [3, 3], padding = 'same', strides = [4, 4])
        #256    -> 512
        self.m3 = Conv2DTranspose(1, kernel_size = [3, 3], padding = 'same', strides = [4, 4])
        #sigmoid
        self.mo = sigmoid()

    def encode(self, inputs):

        #encode 
        x = self.e1(inputs)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)
        return self.e5(x)
    
    def decode(self, inputs):

        #decode
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return self.d5(x)
    
    def pred_mask(self, inputs, training = False):

        #predict mask
        x = self.m1(inputs)
        x = self.m2(x)
        x = self.m3(x)
        if training:
            return x, self.mo(x)
        return self.mo(x)

    def call(self, inputs, mask = True):

        x = self.encode(inputs)
        
        if mask:
            return self.pred_mask(x)            
        return self.decode(x) 

def ae_loss(x, x_pred, mask):

    return tf.math.reduce_mean(tf.math.square(tf.math.multiply((x - x_pred), mask)))

def mask_loss(mask, mask_pred_logits):

    return tf.math.reduce_mean(binary_crossentropy(mask, mask_pred_logits))