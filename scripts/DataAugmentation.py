import numpy as np

from keras import backend as K
from keras.layers import Layer
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps, ImageEnhance


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



def random_distort(image_batch, max_grid_size, max_magnitude):

    """
    Distorts the passed image(s) according to the parameters supplied during
    instantiation, returning the newly distorted image.

    :param images: The image(s) to be distorted.
    :type images: List containing PIL.Image object(s).
    :return: The transformed image(s) as a list of object(s) of type
        PIL.Image.
    """

    def perform_distort(images):

        images = [Image.fromarray(i, "F") for i in images]

        w, h = images[0].size

        horizontal_tiles = np.random.randint(max_grid_size[0]) + 1
        vertical_tiles   = np.random.randint(max_grid_size[1]) + 1

        width_of_square  = int(np.floor(w / float(horizontal_tiles)))
        height_of_square = int(np.floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                        vertical_tile * height_of_square,
                                        width_of_last_square + (horizontal_tile * width_of_square),
                                        height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                        vertical_tile * height_of_square,
                                        width_of_square + (horizontal_tile * width_of_square),
                                        height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                        vertical_tile * height_of_square,
                                        width_of_last_square + (horizontal_tile * width_of_square),
                                        height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                        vertical_tile * height_of_square,
                                        width_of_square + (horizontal_tile * width_of_square),
                                        height_of_square + (height_of_square * vertical_tile)])

        # For loop that generates polygons could be rewritten, but maybe harder to read?
        # polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]

        last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = [[x1, y1, x1, y2, x2, y2, x2, y1] for x1, y1, x2, y2 in dimensions] 

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for a, b, c, d in polygon_indices:
            dx = np.random.randint(-max_magnitude, max_magnitude)
            dy = np.random.randint(-max_magnitude, max_magnitude)

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                            x2, y2,
                            x3 + dx, y3 + dy,
                            x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                            x2 + dx, y2 + dy,
                            x3, y3,
                            x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                            x2, y2,
                            x3, y3,
                            x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                            x2, y2,
                            x3, y3,
                            x4, y4]

        generated_mesh = list(zip(dimensions, polygons))

        # for i, dim in enumerate(dimensions):
        #     generated_mesh.append([dim, polygons[i]])

        augmented_images = [
            np.array(
                image.transform(
                    image.size,
                    Image.MESH, 
                    generated_mesh, 
                    resample=Image.BICUBIC
                )
            ) 
        for image in images]

        return augmented_images

    augmented_batch = np.array([perform_distort(image) for image in image_batch])

    return augmented_batch


class IDGWithLabels():
    def __init__(self, flip=True, rot90=True, distort=True, distort_max_grid_size=(1, 1), distort_max_magnitude=0, **kwargs):
        self.generator = ImageDataGenerator(**kwargs)
        self.flip = flip
        self.rot90 = rot90
        self.distort = distort
        self.distort_max_grid_size = distort_max_grid_size
        self.distort_max_magnitude = distort_max_magnitude
    
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

            if self.distort:
                batch = np.stack((X, y), 1)[...,0]
                batch = random_distort(
                    batch, 
                    max_grid_size = self.distort_max_grid_size,
                    max_magnitude = self.distort_max_magnitude
                )
                X, y = np.moveaxis(batch, 1, 0)

            yield X[..., None], y[..., None]