from tkinter import *
from tkinter import filedialog as fd
from typing import List
import cupy as cp
from numba import jit
import numpy as np
from scipy.fftpack import fft
from scipy.signal.windows import hann


class Calculate:
    """演算用モジュール

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
        """移動平均の計算

        Args:
            data: 二次元行列
            window_size: ウィンドウサイズ
            stride_size: 移動幅サイズ

        Returns:
            移動平均後の二次元行列
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
            x: 一次元データ

        Returns:

        """
        return x / cp.sum(x)

    @staticmethod
    # @jit(nopython=True)
    def renyi_entropy(x: cp.ndarray, r: float) -> float:
        """レニーエントロピーの計算

        Args:
            x: 正規化された信号データ
            r: パラメータ

        Returns:
            レニーエントロピー
        """
        return cp.log(cp.sum(x ** r)) / (1 - r)

    @staticmethod
    def softmax(data: cp.ndarray) -> cp.ndarray:
        """ソフトマックス関数

        Args:
            data: 2次元配列

        Returns:
            ソフトマックス関数を通した2次元配列
        """
        return cp.exp(data) / cp.sum(cp.exp(data), axis=1, keepdims=True)

    @staticmethod
    def stft(x: cp.ndarray, window: int = 100, overlap: int = None) -> cp.ndarray:
        """短時間フーリエ変換

        Args:
            x: 一系列の時系列データ
            window: ウィンドウサイズ
            overlap: オーバーラップサイズ

        Returns:
            短時間フーリエ変換を時間方向に重ねた周波数の分布
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
        """ツァリスエントロピーの計算

        Args:
            x: 正規化された信号データ
            q: パラメータ

        Returns:
            ツァリスエントロピー
        """
        return (1 - cp.sum(x ** q)) / (q - 1)


class FileController:
    """ファイル操作用モジュール

    """

    @staticmethod
    def get_file(file_types: List[str], init_dir: str) -> List[str]:
        """エクスプローラを利用したファイル名の取得

        Args:
            file_types: 取得したいファイルの拡張子リスト
            init_dir: 参照するディレクトリの初期設定

        Returns:
            ファイル名リスト
        """
        root = Tk()
        root.withdraw()
        file_paths = [fd.askopenfilename(filetypes=[[f, f'*.{f}']], initialdir=init_dir) for f in file_types]
        return file_paths
