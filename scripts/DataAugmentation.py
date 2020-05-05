from keras import backend as K
from keras.layers import Layer


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