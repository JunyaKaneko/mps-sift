import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom
import image

__author__ = 'junya@mpsamurai.org'

#
# Octave and its utility functions
#
class Octave:
    def __init__(self, scales, images, sampling_distance):
        self.sampling_distance = sampling_distance
        self.scales = scales
        self.images = images
        self.gradients = np.gradient(self.images, np.power(2, sampling_distance), axis=(1, 2))
        self.orientations = np.arctan2(self.gradients[0], self.gradients[1]) * 180. / np.pi
        self.orientations[self.orientations < 0] += 360.
        self.magnitudes = np.sqrt(np.power(self.gradients[0], 2.) + np.power(self.gradients[1], 2.))
        self.dog_images = np.diff(self.images, axis=0)
        self.dog_scales = self.scales[1:]

    @staticmethod
    def create_from_image(image, sigma, n_divide, sampling_distance):
        scales = np.array([0, ] + [sigma * np.power(2., i / n_divide) for i in range(n_divide + 2)])
        images = np.array([image, ] + [gaussian_filter(image, scale) for scale in scales[1:]])
        return Octave(scales, images, sampling_distance)

    def __iter__(self):
        for i in range(len(self.scales)):
            yield self[i]

    def __getitem__(self, index):
        return {'scale': self.scales[index], 'image': self.images[index],
                'magnitudes': self.magnitudes[index], 'orientations': self.orientations[index]}

    def __len__(self):
        return len(self.images)

    @property
    def shape(self):
        return self.images.shape

    def get_keypoints(self, th):
        keypoints = []
        for i in range(1, len(self.dog_images) - 1):
            for p in image.local_extrema(self.dog_images[i - 1:i + 2]):
                if np.abs(self.dog_images[i][p[0], p[1]]) < th:
                    continue
                else:
                    keypoints.append([self.dog_scales[i], p[0], p[1]])
        return np.array(keypoints)

    def get_keypoint_orientations(self, keypoint, window_shape, th):
        orientations = image.clip(self.get_orientations(keypoint[0]), keypoint[1:], window_shape)
        magnitudes = image.clip(self.get_magnitudes(keypoint[0]), keypoint[1:], window_shape)

        bins = np.linspace(0, 360, 10)
        orientation_histogram = image.orientation_histogram(orientations, magnitudes, bins)
        keypoint_orientations = bins[orientation_histogram > orientation_histogram.max() * th]

        keypoints = []
        for orientation in keypoint_orientations:
            keypoints.append(np.concatenate((keypoint, [orientation, ])))
        return np.array(keypoints)

    def get_descriptor(self, keypoint_orientation, window_shape, divided_window_shape):
        orientations = image.clip(self.get_orientations(keypoint_orientation[0]), keypoint_orientation[1:3], window_shape)
        orientations -= keypoint_orientation[3]
        orientations[orientations < 0] += 360.
        magnitudes = image.clip(self.get_magnitudes(keypoint_orientation[0]), keypoint_orientation[1:3], window_shape)
        
        descriptor = None
        for i in range(0, window_shape[0], divided_window_shape[0]):
            for j in range(0, window_shape[1], divided_window_shape[1]):
                _descriptor = image.orientation_histogram(orientations[i:i + divided_window_shape[0], j:j + divided_window_shape[1]],
                                                          magnitudes[i:i + divided_window_shape[0], j:j + divided_window_shape[1]],
                                                          np.linspace(0, 360, 8))
                if descriptor is None:
                    descriptor = _descriptor
                else:
                    descriptor = np.concatenate((descriptor, _descriptor))
        return descriptor
    
    def get_scale_index(self, scale):
        indices = np.argwhere(self.scales == scale)
        if indices:
            return indices[0][0]
        else:
            None

    def get_image(self, scale):
        index = self.get_scale_index(scale)
        if not index:
            raise ValueError
        else:
            return self.images[index]

    def get_orientations(self, scale):
        index = self.get_scale_index(scale)
        if not index:
            raise ValueError
        else:
            return self.orientations[index]

    def get_magnitudes(self, scale):
        index = self.get_scale_index(scale)
        if not index:
            raise ValueError
        else:
            return self.magnitudes[index]

    def get_pixel(self, coordinate):
        index = self.get_scale_index(coordinate[0])
        if not index:
            raise ValueError
        else:
            return image[index, coordinate[1], coordinate[2]]
        

#
# Sacle Space
#
class ScaleSpace:
    def __init__(self, octaves):
        self.octaves = octaves

    @staticmethod
    def create_from_image(image, sigma, n_divide, n_octave):
        octaves = []
        base_image = image
        for i in range(n_octave):
            octaves.append(Octave.create_from_image(base_image, sigma, n_divide, np.power(2., i)))
            base_image = octaves[-1][n_divide]['image'][::2, ::2]
        return ScaleSpace(octaves)

    def get_keypoints(self, th):
        keypoints = None
        for i, octave in enumerate(self.octaves):
            keypoints_in_octave = np.array([np.concatenate(([i,], k)) for k in octave.get_keypoints(th)])
            if keypoints is None and len(keypoints_in_octave):
                keypoints = keypoints_in_octave
            elif len(keypoints_in_octave):
                keypoints = np.concatenate((keypoints, keypoints_in_octave))
        return keypoints

    def get_keypoint_orientations(self, keypoints, window_shape, th):
        keypoint_orientations = []
        for keypoint in keypoints:
            for keypoint_orientation in self[keypoint[0]].get_keypoint_orientations(keypoint[1:], window_shape, th):
                keypoint_orientations.append(np.concatenate(([keypoint[0],], keypoint_orientation)))
        return np.array(keypoint_orientations)

    def get_descriptors(self, keypoint_orientations, window_shape, divided_window_shape):
        descriptors = []
        for keypoint_orientation in keypoint_orientations:
            descriptors.append(
                self[keypoint_orientation[0]].get_descriptor(keypoint_orientation[1:], 
                                                             window_shape, divided_window_shape))
        return np.array(descriptors)

                
    def apply(self, func, *args, **kwargs):
        octaves = [func(octave, *args, **kwargs) for octave in self]
        return ScaleSpace(octaves)

    def get_image(self, octave_index, scale):
        return self[octave_index].get_image(scale)

    def get_orientations(self, octave_index, scale):
        return self[octave_index].get_orientations(scale)

    def get_magnitudes(self, octave_index, scale):
        return self[octave_index].get_magnitudes(scale)

    def get_pixel(self, coordinate):
        return self[coordinate[0]].get_pixel(coordinate[1:])

    def __iter__(self):
        for octave in self.octaves:
            yield octave

    def __len__(self):
        return len(self.octaves)

    def __getitem__(self, index):
        if isinstance(index, (np.float64, np.float32)):
            index = int(index)
        return self.octaves[index]


if __name__ == '__main__':
    from scipy.ndimage import imread
    from matplotlib import pyplot as plt

    scale, n_divide, n_octave, th = 1.2 ,3, 3, 0.02    
    test_image = image.normalize(gaussian_filter(imread('../images/junya.png', flatten=True), 1.0))
    
    scalespace = ScaleSpace.create_from_image(test_image, scale, n_divide, n_octave)
    keypoints = scalespace.get_keypoints(th)
    keypoint_orientations = scalespace.get_keypoint_orientations(keypoints, (16, 16), 0.8)
    descriptors = scalespace.get_descriptors(keypoint_orientations, (16, 16), (4, 4))

    print(descriptors)

    fig, axes = plt.subplots(len(scalespace), len(scalespace[0]))
    for i, row_axes in enumerate(axes):
        for j, axis in enumerate(row_axes):
            scale = scalespace[i][j]['scale']
            axis.imshow(scalespace[i][j]['image'])
            for keypoint in keypoint_orientations:
                if keypoint[1] == scale:
                    circle = plt.Circle((keypoint[3], keypoint[2]), 5.0 / np.power(2, i), color='r')
                    axis.add_artist(circle)
                    axis.arrow(keypoint[3], keypoint[2], 
                               (16 / np.power(2, i)) * np.cos(keypoint[4] * np.pi / 180.),
                               (16 / np.power(2, i)) * np.sin(keypoint[4] * np.pi / 180.),
                               head_width=8. / np.power(2, i), head_length=8. / np.power(2, i), 
                               fc='k', ec='k')
        
    fig, axes = plt.subplots(len(scalespace), len(scalespace[0]))
    for i, row_axes in enumerate(axes):
        for j, axis in enumerate(row_axes):
            scale = scalespace[i][j]['scale']
            axis.imshow(scalespace[i][j]['magnitudes'])
            for keypoint in keypoints:
                if keypoint[1] == scale:
                    circle = plt.Circle((keypoint[3], keypoint[2]), 5.0 / np.power(2, i), color='r')
                    axis.add_artist(circle)

    fig, axes = plt.subplots(len(scalespace), len(scalespace[0]))
    for i, row_axes in enumerate(axes):
        for j, axis in enumerate(row_axes):
            scale = scalespace[i][j]['scale']
            axis.imshow(scalespace[i][j]['orientations'])
            for keypoint in keypoints:
                if keypoint[1] == scale:
                    circle = plt.Circle((keypoint[3], keypoint[2]), 5.0 / np.power(2, i), color='r')
                    axis.add_artist(circle)

    #for keypoint in keypoints:
    #    descriptor = scalespaceget_descriptor(keypoint[1:], (16, 16), (4, 4))
    #    print(len(descriptor))

    plt.show()
