import numpy as np
from scipy import signal


__author__ = 'junya@mpsamurai.org'


def normalize(image):
    '''Normalize image so that the image has values from 0 to 1'''
    return (image - image.min()) / (image - image.min()).max()


def local_extrema(tensor):
    '''Get Local extrema from (3, row, col) tensor'''
    assert tensor.shape[0] == 3

    def is_extremum(cr, cc):
        return tensor[1, cr, cc] <= np.min(tensor[:, cr-1:cr+2, cc-1:cc+2]) or \
            tensor[1, cr, cc] >= np.max(tensor[:, cr-1:cr+2, cc-1:cc+2])        

    _, n_rows, n_cols = tensor.shape
    return [[cr, cc] 
            for cr in range(1, n_rows -1) for cc in range(1, n_cols - 1) if is_extremum(cr, cc)]


def gaussian_window(size, sigma):
    window_1d = signal.gaussian(size, sigma)
    window_2d = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            window_2d[i, j] = window_1d[i] * window_1d[j]
    return window_2d


def clip(image, center, shape):
    n_clipped_row, n_clipped_col = shape
    clipped_cr, clipped_cc = n_clipped_row // 2 - 1, n_clipped_col // 2 - 1

    cr, cc = center

    n_padding_row, n_padding_col = 0, 0
    if clipped_cr - cr > 0:
        n_padding_row = clipped_cr - cr
    if clipped_cc - cc > 0:
        n_padding_col = clipped_cc - cc

    clipped_image = np.zeros(shape=shape)
    for r in range(n_clipped_row - n_padding_row):
        for c in range(n_clipped_col - n_padding_col):
            if cr - clipped_cr + n_padding_row + r < image.shape[0] \
                    and cc - clipped_cc + n_padding_col + c < image.shape[1]: 
                clipped_image[n_padding_row + r, n_padding_col + c] \
                    = image[cr - clipped_cr + n_padding_row + r, 
                            cc - clipped_cc + n_padding_col + c]
    return clipped_image


def orientation_histogram(orientations, magnitudes, bins):
    orientations = orientations.flatten()
    magnitudes = magnitudes.flatten()

    histogram = np.zeros(shape=(len(bins)))
    for i in range(len(orientations) - 1):
        for j in range(len(bins)):
            if bins[j] <= orientations[i]  < bins[j + 1]:
                histogram[j] += magnitudes[j]

    return histogram
