from tkinter import *
from tkinter import filedialog as fd
from typing import List
import cupy as cp
from numba import jit
import numpy as np
from scipy.fftpack import fft
from scipy.signal.windows import hann


class Calculate:
    """Operation modules

    """

    @staticmethod
    # @jit(nopython=True)
    def accuracy_rate(output: cp.ndarray, train_label: cp.ndarray, N: int) -> float:
        return cp.dot(output, train_label) / N

    @staticmethod
    # @jit(nopython=True)
    def cross_entropy(data: cp.ndarray, label: cp.ndarray) -> cp.ndarray:
        data_size, _ = data.shape
        return - cp.sum(cp.log(cp.sum(data * label, axis=1))) / data_size

    @staticmethod
    # @jit(nopython=True)
    def moving_average(data: cp.ndarray, window_size: int, stride_size: int) -> cp.ndarray:
        """Calculation of moving averages

        Args:
            data: two-dimensional matrix
            window_size: window size
            stride_size: Moving width size

        Returns:
            2D matrix after moving average
        """
        data_length, res_size = data.shape
        averaged_length = (data_length - window_size) // stride_size + 1
        averaged_data = cp.zeros((averaged_length, res_size))
        for i in range(0, averaged_length):
            averaged_data[i] = cp.sum(data[i * stride_size:i * stride_size + window_size], axis=0) / window_size
        return averaged_data

    @staticmethod
    # @jit(nopython=True)
    def normalize(x: cp.ndarray) -> cp.ndarray:
        """

        Args:
            x: one-dimensional data

        Returns:

        """
        return x / cp.sum(x)

    @staticmethod
    # @jit(nopython=True)
    def renyi_entropy(x: cp.ndarray, r: float) -> float:
        """Calculation of Reny-Entropy

        Args:
            x: normalized signal data
            r: parameter

        Returns:
            Renyi entropy
        """
        return cp.log(cp.sum(x ** r)) / (1 - r)

    @staticmethod
    def softmax(data: cp.ndarray) -> cp.ndarray:
        """softmax function

        Args:
            data: two-dimensional array

        Returns:
            2D arrays through softmax functions
        """
        return cp.exp(data) / cp.sum(cp.exp(data), axis=1, keepdims=True)

    @staticmethod
    def stft(x: cp.ndarray, window: int = 100, overlap: int = None) -> cp.ndarray:
        """short time Fourier transform

        Args:
            x: One series of time series data
            window: window size
            overlap: overlap size

        Returns:
            Distribution of frequencies superimposed on the short-time Fourier transform in the time direction
        """
        if not overlap:
            overlap = window // 2
        default_length = x.size
        stft_length = (default_length - window) // overlap + 1
        segmented_amps = cp.array(
            [cp.abs(fft(x[i * overlap:i * overlap + window] * hann(window))) / (window / 2) for i in
             range(stft_length)])
        res = cp.sum(segmented_amps, axis=0)
        return res

    @staticmethod
    # @jit(nopython=True)
    def tsallis_entropy(x: cp.ndarray, q: float) -> float:
        """Calculation of Tsallis entropy

        Args:
            x: normalized signal data
            q: parameter

        Returns:
            Tsaris entropy
        """
        return (1 - cp.sum(x ** q)) / (q - 1)


class FileController:
    """ Module for file manipulation

    """

    @staticmethod
    def get_file(file_types: List[str], init_dir: str) -> List[str]:
        """Obtaining file names using Explorer

        Args:
            file_types: List of file extensions to be retrieved
            init_dir: Initial settings for directories to be referenced

        Returns:
            filename list
        """
        root = Tk()
        root.withdraw()
        file_paths = [fd.askopenfilename(filetypes=[[f, f'*.{f}']], initialdir=init_dir) for f in file_types]
        return file_paths
