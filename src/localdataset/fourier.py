'''
Module provides fourier transformation functions used to determine the relevance of ECG signals
'''

from scipy.fft import fft, rfft, fftfreq
import numpy as np


def real_fourier_transformation(sample: np.ndarray, sampling_rate: int):
    N = sample.shape[0]
    normalize = N/2
    fourier = rfft(sample)
    requency_axis = fftfreq(N, d=1.0/sampling_rate)
    norm_amplitude = np.abs(fourier)/normalize
    return requency_axis[:int(np.ceil(N/2))], norm_amplitude


def fourier_transformation(sample: np.ndarray, sampling_rate: int):
    N = sample.shape[0]
    normalize = N/2
    fourier = fft(sample)
    requency_axis = fftfreq(N, d=1.0/sampling_rate)
    norm_amplitude = np.abs(fourier)/normalize
    return requency_axis[:int(np.ceil(N/2))], norm_amplitude[:int(np.ceil(N/2))]


def normalized_fourier_transformation(sample: np.ndarray, sampling_rate: int, bins=np.arange(0, 210, 1)):
    result = fourier_transformation(sample, sampling_rate)
    print(result)
    return np.histogram(result[1], bins=bins, weights=result[0])


def fourier_transformation_normalized_avg_histogram(sample: np.ndarray, sampling_rate: int, bins=np.arange(0, 150.1, 1)):
    data = fourier_transformation(sample, sampling_rate)
    hist = np.histogram(data[0], bins=bins, weights=data[1])
    inds = np.digitize(data[0], hist[1])
    samples_in_bins = np.bincount(inds, minlength=hist[1].shape[0])
    return hist[0] / samples_in_bins[1:]
