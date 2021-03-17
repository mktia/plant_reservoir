from datetime import datetime as dt
from tkinter import *
from tkinter import filedialog as fd
from typing import List, Tuple
import warnings
from numba import jit
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.fftpack import fft
from scipy import linalg, stats
from scipy.signal.windows import hann
from scipy.spatial.distance import cdist, euclidean
from scipy.optimize import linear_sum_assignment


class Calculate:
    """演算用モジュール

    """

    @staticmethod
    @jit(nopython=True)
    def accuracy_rate(output: np.ndarray, train_label: np.ndarray, N: int) -> float:
        return np.dot(output, train_label) / N

    @staticmethod
    def between_centers_of_gravity(data_1: np.ndarray, data_2: np.ndarray) -> float:
        cog_1 = Calculate.center_of_gravity(data_1)
        cog_2 = Calculate.center_of_gravity(data_2)
        return euclidean(cog_1, cog_2)

    @staticmethod
    def center_of_gravity(data: np.ndarray) -> np.ndarray:
        return np.mean(data, axis=0)

    @staticmethod
    @jit(nopython=True)
    def cross_entropy(data: np.ndarray, label: np.ndarray) -> np.ndarray:
        data_size, _ = data.shape
        return - np.sum(np.log(np.sum(data * label, axis=1))) / data_size

    @staticmethod
    def emd(data_1: np.ndarray, data_2: np.ndarray) -> float:
        # time size x number of feature points
        # size, _ = data_1.shape
        size = 1000
        d = cdist(data_1, data_2)
        # d = fastdist.matrix_to_matrix_distance(data_1, data_2, fastdist.euclidean, 'euclidean')
        assignment = linear_sum_assignment(d)
        # return d[assignment].sum() / size / np.var(d)
        return d[assignment].sum() / size

    @staticmethod
    def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, List[float]]:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        print(f'slope: {slope}, correlation: {r_value}, p-value: {p_value}, standard error: {std_err}')
        return slope, intercept, [r_value, p_value, std_err]

    @staticmethod
    @jit(nopython=True)
    def moving_average(data: np.ndarray, window_size: int, stride_size: int) -> np.ndarray:
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
        averaged_data = np.zeros((averaged_length, res_size))
        for i in range(0, averaged_length):
            averaged_data[i] = np.sum(data[i * stride_size:i * stride_size + window_size], axis=0) / window_size
        return averaged_data

    @staticmethod
    def nearest(data_1: np.ndarray, data_2: np.ndarray) -> float:
        distances = cdist(data_1, data_2)
        return np.min(distances)

    @staticmethod
    @jit(nopython=True)
    def normalize(x: np.ndarray) -> np.ndarray:
        """

        Args:
            x: 一次元データ

        Returns:

        """
        # todo: fix
        return x / np.sum(x)

    @staticmethod
    @jit(nopython=True)
    def renyi_entropy(x: np.ndarray, r: float) -> float:
        """レニーエントロピーの計算

        Args:
            x: 正規化された信号データ
            r: パラメータ

        Returns:
            レニーエントロピー
        """
        return np.log(np.sum(x ** r)) / (1 - r)

    @staticmethod
    def ridge_regression(X: np.ndarray, Y: np.ndarray, reg=1e6) -> np.ndarray:
        """リッジ回帰の計算

        Args:
            X: 学習データ
            Y: ラベル
            reg: 正則化パラメータ

        Returns:
            リッジ回帰によって得られた重み
        """
        _, eye_size = X.shape
        return np.dot(np.dot(Y.T, X), linalg.inv(np.dot(X.T, X) + reg * np.eye(eye_size)))

    @staticmethod
    def softmax(data: np.ndarray) -> np.ndarray:
        """ソフトマックス関数

        Args:
            data: 2次元配列

        Returns:
            ソフトマックス関数を通した2次元配列
        """
        return np.exp(data) / np.sum(np.exp(data), axis=1, keepdims=True)

    @staticmethod
    def std(data: np.ndarray) -> np.ndarray:
        """標準偏差

        Args:
            data:

        Returns:

        """
        return np.std(data, axis=0)

    @staticmethod
    def stft(x: np.ndarray, window: int = 100, overlap: int = None) -> np.ndarray:
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
        segmented_amps = np.array(
            [np.abs(fft(x[i * overlap:i * overlap + window] * hann(window))) / (window / 2) for i in
             range(stft_length)])
        res = np.sum(segmented_amps, axis=0)
        return res

    @staticmethod
    def sum_sq_from_center_of_gravity(data):
        size, _ = data.shape
        cog = Calculate.center_of_gravity(data)
        d = 0
        for i in range(size):
            d += np.square(euclidean(cog, data[i]))
        return d

    @staticmethod
    @jit(nopython=True)
    def tsallis_entropy(x: np.ndarray, q: float) -> float:
        """ツァリスエントロピーの計算

        Args:
            x: 正規化された信号データ
            q: パラメータ

        Returns:
            ツァリスエントロピー
        """
        return (1 - np.sum(x ** q)) / (q - 1)

    @staticmethod
    def ward(data_1: np.ndarray, data_2: np.ndarray) -> float:
        # sum_square_1 = Calculate.sum_sq_from_center_of_gravity(data_1)
        # sum_square_2 = Calculate.sum_sq_from_center_of_gravity(data_2)
        # sum_square_all = Calculate.sum_sq_from_center_of_gravity(np.vstack((data_1, data_2)))
        # return sum_square_all - (sum_square_1 + sum_square_2)
        len_1, _ = data_1.shape
        len_2, _ = data_2.shape
        cog_1 = Calculate.center_of_gravity(data_1)
        cog_2 = Calculate.center_of_gravity(data_2)
        d = len_1 * len_2 / (len_1 + len_2) * np.square(euclidean(cog_1, cog_2))
        return d


class FileController:
    """ファイル操作用モジュール

    """

    @staticmethod
    def export_data(data: np.ndarray, file_name: str, dir: str = './data_out', title: str = ''):
        """日付，ファイル名をつけて保存

        Args:
            data: データ
            file_name: 使用したファイル名
            dir: 出力ディレクトリ
            title: 出力ファイル名

        Returns:
            None
        """
        date_label = dt.today().isoformat().replace(':', '-').split('.')[0]
        prev_title = f'{date_label}_{file_name.split("/")[-1][:-4]}'
        np.savetxt(f'{dir}/{prev_title}_{title}.csv', data, delimiter=',', fmt='%.6f')
        print(f'saved: {prev_title}_{title}.csv')

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


class GraphViewer:
    """グラフ出力用モジュール

    """

    warnings.simplefilter('ignore')

    def __init__(self, fig_size: List[float] = None, font_size: int = 18, title_size: int = 22, legend_size: int = 18,
                 tick_size: int = 16, is_constrained=True):
        if fig_size == None:
            self.__figure_size = [6.4, 4.8]
        else:
            self.__figure_size = fig_size
        self.__font_size = font_size
        self.__legend_size = legend_size
        self.__title_size = title_size
        self.__tick_size = tick_size

        # np.random.seed(32)
        # self.__color = np.random.randint(0, 255, (100, 3)) / 255

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Hervetica']
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['figure.constrained_layout.use'] = is_constrained

        self.__prev_title = dt.today().isoformat().replace(':', '-').split('.')[0]

    def bar(self, data: np.ndarray or str, save_as: str, title: str = '',
            xlabel: str = '', ylabel: str = ''):
        fig = plt.figure(figsize=self.__figure_size)
        ax = fig.add_subplot()

        y_size, x_size = data.shape
        for y in range(y_size):
            ax.bar(np.arange(x_size), data[y])
        ax.set_title(title, fontsize=self.__title_size)
        ax.set_xlabel(xlabel, fontsize=self.__font_size)
        ax.set_ylabel(ylabel, fontsize=self.__font_size)
        ax.tick_params(labelsize=self.__tick_size)

        self.save_as_eps(fig, save_as)

    def boxplot(self, data: List[np.ndarray], labels: list, save_as: str, title: str = '', xlabel: str = '',
                ylabel: str = ''):
        fig = plt.figure(figsize=self.__figure_size)
        ax = fig.add_subplot()

        ax.boxplot(data, notch=False, labels=labels, sym='+', medianprops=dict(linewidth=2))
        ax.set_title(title, fontsize=self.__title_size)
        ax.set_xlabel(xlabel, fontsize=self.__font_size)
        ax.set_ylabel(ylabel, fontsize=self.__font_size)
        ax.tick_params(labelsize=self.__tick_size)

        self.save_as_eps(fig, save_as)

    def line(self, data: np.ndarray or str, save_as: str, x_data: np.ndarray = None, line_width=0.5, title: str = '',
             xlabel: str = '', ylabel: str = ''):
        fig = plt.figure(figsize=self.__figure_size)
        ax = fig.add_subplot()

        cm = plt.get_cmap()

        y_size, x_size = data.shape
        for y in range(y_size):
            if x_data is None:
                ax.plot(x_data, data[y], linewidth=line_width)
            else:
                ax.plot(data[y], linewidth=line_width)
        ax.set_title(title, fontsize=self.__title_size)
        ax.set_xlabel(xlabel, fontsize=self.__font_size)
        ax.set_ylabel(ylabel, fontsize=self.__font_size)
        ax.tick_params(labelsize=self.__tick_size)

        self.save_as_eps(fig, save_as)

    def line_with_error(self, data: np.ndarray or str, x_data: np.ndarray, err_data: np.ndarray, save_as: str,
                        title: str = '', xlabel: str = '', ylabel: str = '', legends: list = [], line_width=0.5):
        fig = plt.figure(figsize=self.__figure_size)
        ax = fig.add_subplot()

        y_size, x_size = data.shape
        for y in range(y_size):
            ax.plot(x_data, data[y], color=f'C{y}', linewidth=line_width)
            fill_top = np.min(np.vstack((data[y] + err_data[y], np.ones(data[y].size))), axis=0)
            fill_bottom = data[y] - err_data[y]
            ax.fill_between(x_data, fill_top, fill_bottom, color=f'C{y}', alpha=0.3)
        ax.set_title(title, fontsize=self.__title_size)
        ax.set_xlabel(xlabel, fontsize=self.__font_size)
        ax.set_ylabel(ylabel, fontsize=self.__font_size)
        ax.tick_params(labelsize=self.__tick_size)
        if len(legends) != 0:
            ax.legend(legends, loc='lower right', fontsize=self.__font_size)

        # self.save_as_eps(fig, save_as)

        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(f'./image_out/{self.__prev_title}_{save_as}.pdf') as pdf:
            pdf.savefig(fig)

    def scatter(self, xs: np.ndarray, ys: np.ndarray, save_as: str, is_reg=True, slope=0, intercept=0,
                legend: list = None, legend_cols: int = 1, title: str = '', xlabel: str = '',
                ylabel: str = ''):
        fig = plt.figure(figsize=self.__figure_size)
        ax = fig.add_subplot()

        if xs.ndim != 1:
            x_size, _ = xs.shape
            for i in range(x_size):
                ax.scatter(xs[i], ys[i], color=f'C{i}', marker='x')
        else:
            ax.scatter(xs, ys, color=f'C0', marker='x')

        if legend is not None:
            ax.legend(legend, ncol=legend_cols)

        if is_reg:
            ax.plot(xs.flatten(), slope * xs.flatten() + intercept, color='C9')

        ax.set_title(title, fontsize=self.__title_size)
        ax.set_xlabel(xlabel, fontsize=self.__font_size)
        ax.set_ylabel(ylabel, fontsize=self.__font_size)
        ax.tick_params(labelsize=self.__tick_size)

        self.save_as_eps(fig, save_as)

    def scatter3d_each_class(self, data: np.ndarray, units_idx: list, save_as: str, T: int = 8000, class_size: int = 6,
                             is_reg=True, slope=0, intercept=0, legend: list = None, legend_cols: int = 1,
                             title: str = '', xlabel: str = '', ylabel: str = '', zlabel: str = ''):
        fig = plt.figure(figsize=self.__figure_size)
        # ax = fig.add_subplot(projection='3d')
        ax = Axes3D(fig)

        u1, u2, u3 = units_idx
        for cls in range(class_size):
            ax.scatter(data[cls * T:(cls + 1) * T, u1], data[cls * T:(cls + 1) * T, u2],
                       data[cls * T:(cls + 1) * T, u3], color=f'C{cls}', marker='.', s=5)

        ax.set_title(title, fontsize=self.__title_size)
        ax.set_xlabel(xlabel, fontsize=self.__font_size, labelpad=20)
        ax.set_ylabel(ylabel, fontsize=self.__font_size, labelpad=20)
        ax.set_zlabel(zlabel, fontsize=self.__font_size, labelpad=20)
        ax.legend([f'class {i}' for i in range(class_size)], fontsize=self.__legend_size, loc='upper left',
                  frameon=False, markerscale=5)
        ax.tick_params(labelsize=self.__tick_size, direction='in')
        ax.view_init(elev=36,azim=320)
        for idx, l in enumerate(ax.get_xticklabels()):
            if idx % 2 == 0:
                l.set_visible(False)
        for idx, l in enumerate(ax.get_yticklabels()):
            if idx % 2 == 0:
                l.set_visible(False)
        for idx, l in enumerate(ax.get_zticklabels()):
            if idx % 2 == 0:
                l.set_visible(False)

        plt.show()

        self.save_as_eps(fig, save_as)

    def save_as_eps(self, fig: plt.Figure, title: str, save_dir: str = './image_out'):
        fig.savefig(f'{save_dir}/{self.__prev_title}_{title}.eps', format='eps', dpi=300)


class MatplotlibExtension:
    @staticmethod
    def get_custom_cmap():
        tab20b = plt.get_cmap('tab20b').colors
        tab20c = plt.get_cmap('tab20c').colors
        default_colors = tab20c + tab20b
        new_colors = [default_colors[i + 2 * j] for i in range(2) for j in range(20)]

        return LinearSegmentedColormap.from_list('custom_cmap', new_colors, N=len(new_colors))
