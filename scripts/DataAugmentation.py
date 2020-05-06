import numpy as np

from keras import backend as K
from keras.layers import Layer
from keras.preprocessing.image import ImageDataGenerator

class AdditiveSPNoise(Layer):

    def __init__(self, p, max_amp, **kwargs):
        self.p = p
        self.max_amp = max_amp
        super(AdditiveSPNoise, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        def noised():
            shp    = K.shape(inputs)
            salt   = K.random_binomial(shape=shp, p=self.p)
            pepper = K.random_binomial(shape=shp, p=self.p)
            amp    = K.random_uniform(shape=shp[0:1], minval=0, maxval=self.max_amp)
            amp    = K.reshape(amp, [shp[0], 1, 1, 1])
            
            out = inputs + amp * (salt - pepper)
            return out

        return K.in_train_phase(noised(), inputs, training=training)


class AdditiveGaussianNoise(Layer):

    def __init__(self, sigma, epsilon, **kwargs):
        self.sigma = sigma
        self.epsilon = epsilon
        super(AdditiveGaussianNoise, self).__init__(**kwargs)
        
    def call(self, inputs, training=None):       
        def noised():
            shp    = K.shape(inputs)       
            noise  = K.random_normal(shape=shp, stddev=self.sigma)
            amp    = K.random_uniform(shape=shp[0:1], minval=0, maxval=self.epsilon)
            amp    = K.reshape(amp, [shp[0], 1, 1, 1])
            
            out = inputs + amp * noise
            return out

        return K.in_train_phase(noised(), inputs, training=training)


class IDGWithLabels():
    def __init__(self, flip=True, rot90=True, **kwargs):
        self.generator = ImageDataGenerator(**kwargs)
        self.flip = flip
        self.rot90 = rot90
    
    def flow(self, *args, **kwargs):
        for X, y in self.generator.flow(*args, **kwargs):
            if self.flip:
                k = np.random.binomial(1, 0.5, size=2) * 2 - 1
                X = X[:, ::k[0], ::k[1]]
                y = y[:, ::k[0], ::k[1]]
                
            if self.rot90:
                k = np.random.randint(4)
                X = np.rot90(X, k, (1, 2))
                y = np.rot90(y, k, (1, 2))

            yield X, y