from datetime import date
from typing import List, Set, Tuple, Union
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from scipy import fftpack, linalg, stats
from tqdm import tqdm
from utils import Calculate, FileController


def classify(output: np.ndarray) -> np.ndarray:
    return np.argmax(output, axis=1)


@jit(nopython=True)
def correlation_integral(distance_X: np.ndarray, r: float) -> int:
    """相関積分を求める

    Args:
        distance_X: 各点の距離 (N x N-1)
        r: 超球の半径

    Returns:
        相関積分の値
    """
    N, _ = distance_X.shape
    diff = r - distance_X
    return np.count_nonzero(diff >= 0) / N ** 2


def corr_integral_of_each_params(distance_X: np.ndarray, r_params: np.ndarray, units_size) -> np.ndarray:
    Cs = np.array([correlation_integral(distance_X, r) for r in r_params])
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(np.log(r_params), np.log(Cs))
    set_graph_params(ax, f'log C(r) / log r [{units_size} units]', 'log r', 'log C(r)', 'x-large', 'x-large', 'large')
    plt.show()
    return Cs


@jit(nopython=True)
def distance(X: np.ndarray) -> np.ndarray:
    """各点の差の絶対値による距離を求める

    Args:
        X: 時系列データ (Nステップ x m次元)

    Returns:
        各点間の距離の行列
    """
    N, _ = X.shape
    distance_X = np.zeros((N, N - 1))
    for i in range(N):
        diff_xs = X[i, :] - np.vstack((X[:i, :], X[i + 1:, :]))
        distance_X[i, :] = np.sum(np.abs(diff_xs), axis=1)
    return distance_X


@jit(nopython=True)
def dot(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.dot(X, Y)


def get_random_units(random_size: int, num_of_select_units: int, removed_units: Set[int]) -> List[int]:
    """指定した数のユニットのインデックスをランダムに取得する

    Args:
        random_size: 取りうるユニット数
        num_of_select_units: 選択するユニットの数
        removed_units: 取り除くユニットのインデックス

    Returns:
        ランダムに取得したインデックスリスト
    """
    units = None
    while True:
        units = set(np.random.randint(0, random_size, num_of_select_units))
        if len(units) == units_size and len(removed_units & units) == 0:
            break
    return list(units)


def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, List[float]]:
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f'slope: {slope}, correlation: {r_value}, p-value: {p_value}, standard error: {std_err}')
    return slope, intercept, [r_value ** 2, p_value, std_err]


def ridge_regression(data: np.ndarray, label: np.ndarray) -> np.ndarray:
    reg = 1e4
    _, eye_size = data.shape
    return np.dot(np.dot(label, data), linalg.inv(np.dot(data.T, data) + reg * np.eye(eye_size)))


def set_graph_params(ax, title: str, xlabel: str, ylabel: str, title_size: Union[str, float, int],
                     font_size: Union[str, float, int], label_size: Union[str, float, int]) -> None:
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(labelsize=label_size)


if __name__ == '__main__':
    file_types = ['csv']
    init_dir = './data_in'
    coords_file, label_file = './data_in/0513_l123r123_q1_md20_ws15_fixed.csv', './data_in/1111_l123r123_q1_md20_ws15_label.csv'
    # coords_file, label_file = FileController.get_file(file_types * 2, init_dir)
    coords_data = np.loadtxt(coords_file, delimiter=',')
    label_data = np.loadtxt(label_file, delimiter=',')
    print(f'coords: {coords_file.split("/")[-1]}, label: {label_file.split("/")[-1]}')

    T = 8000
    test_T = 500
    class_size = 6
    time_size, reservoir_size = coords_data.shape

    # l123r123
    coords_data = np.vstack((coords_data[:9315], coords_data[27585:36900], coords_data[9315:18360],
                             coords_data[36900:46245], coords_data[18360:27585], coords_data[46245:]))
    # end_frames = [9315, 18360, 27585, 36900, 46245, 55830]
    # lr1lr2lr3
    end_frames = [9315, 18630, 27675, 37020, 46245, 55830]

    # data preprocessing
    X = np.ones((1, reservoir_size))
    test_data = np.copy(X)
    # std_X = np.copy(X)
    label = np.ones((1, class_size))
    test_label = np.copy(label)
    start = 0
    for end in end_frames:
        X = np.vstack((X, coords_data[start:start + T]))
        test_data = np.vstack((test_data, coords_data[start + T:start + T + test_T]))
        label = np.vstack((label, label_data[start:start + T]))
        test_label = np.vstack((test_label, label_data[start + T:start + T + test_T]))
        start = end
    X = X[1:]
    test_data = test_data[1:]
    label = label[1:]
    test_label = test_label[1:]
    #
    # for i in range(class_size):
    #     std_X = np.vstack((std_X, coords_data[i * T:(i + 1) * T] - np.average(coords_data[i * T:(i + 1) * T], axis=0)))
    # std_X = std_X[1:]

    # regularization coefficient
    reg = 1e4
    Yt_T = label.T
    stride, window = 1, 11
    # sampling size
    stft_N = 300
    # sampling interval = 1 / sampling frequency
    dt = 1e-2
    # nyquist frequency
    # nyquist = 1 / dt / 2
    # cut before this frequency
    # high_pass = 1
    # low_pass = 30
    # stft_freq = fftpack.fftfreq(stft_N, dt)

    np.random.seed(32)

    # ignore units
    ignore_indices = set([i for i, x in enumerate(X[0, :]) if x == 0])
    print(f'ignore indices: {sorted(list(ignore_indices))}')

    units_sizes = [3, 5, 10, 20]
    # units_sizes = [3, 5, 10]
    # units_sizes = [int(i + 1) for i in range(10)]
    units_len = len(units_sizes)
    # test 100 times
    steps = 100
    test_length = (test_T * class_size - window) // stride + 1
    classified = np.zeros((steps, test_length, len(units_sizes)))
    distributions = np.zeros((steps, stft_N, len(units_sizes)))
    averaged_test_label = test_label[5:-5]
    classified_label = np.argmax(averaged_test_label, axis=1)
    # entropy parameters
    # t_param, r_param = [float(i) for i in input('param t & r: ').split(' ')]
    # records = []
    rad_params = np.array([10 ** (- num + 2) for num in range(5)][::-1])
    log_rad_params = np.log(rad_params)
    accs, errors, dims = [[] for i in range(3)]
    # accs, S, t_S, r_S, errors = [[] for i in range(5)]
    units_combinations=[]
    for k, units_size in enumerate(units_sizes):
        for count in tqdm(range(steps)):
            random_indices = get_random_units(reservoir_size, units_size, ignore_indices)
            units_combinations.append(random_indices)
    #         X_selected = np.hstack([X[:, i][:, np.newaxis] for i in random_indices])
    #         Wout = ridge_regression(X_selected, Yt_T)
    #         length = (test_T * class_size - window) // stride + 1
    #         averaged_test_data = Calculate.moving_average(test_data, window, stride)
    #         averaged_test_data_selected = np.hstack(
    #             [averaged_test_data[:, i][:, np.newaxis] for i in random_indices])
    #
    #         distance_X = distance(X_selected)
    #         C = corr_integral_of_each_params(distance_X, rad_params, units_size)
    #         C = np.log(C)
    #         x = np.array([log_rad_params[i + 1] - log_rad_params[i] for i in range(len(C) - 1)])
    #         y = np.array([C[i + 1] - C[i] for i in range(len(C) - 1)])
    #         dims.append(np.max(y / x))
    #
    #         # Y = np.dot(Wout, averaged_test_data_selected.T).T
    #         Y = dot(Wout, averaged_test_data_selected.T).T
    #         # classification
    #         classified[count, :, k] = classify(Y)
    #
    #         Y_softmax = Calculate.softmax(Y)
    #         # error
    #         cross_entropy = Calculate.cross_entropy(Y_softmax, averaged_test_label)
    #         errors.append(cross_entropy)
    #
    #         # calculate accuracy rate
    #         wrong_count = np.count_nonzero(classified[count, :, k] - classified_label)
    #         accuracy_rate = 1 - wrong_count / length
    #         accs.append(accuracy_rate)
    #
    #         # signals = np.zeros((units_size, stft_N))
    #         # for i, idx in enumerate(random_indices):
    #         #     amp = Calculate.stft(std_X[:, idx], stft_N)
    #         #     amp[stft_freq < high_pass] = 0
    #         #     amp[stft_freq > low_pass] = 0
    #         #     power = amp ** 2
    #         #     signals[i, :] = Calculate.normalize(power)
    #
    #         # integrate and normalize series of each unit
    #         # integrated = Calculate.normalize(np.max(signals, axis=0))
    #         # distributions[count, :, k] = integrated
    #
    #         # x = integrated[high_pass * 3:stft_N // 2]
    #         # x = integrated[high_pass * 3:low_pass * 3]
    #         # S.append(stats.entropy(x))
    #         # t_S.append(Calculate.tsallis_entropy(x, t_param))
    #         # r_S.append(Calculate.renyi_entropy(x, r_param))
    #
    #         # records.append(random_indices + [accuracy_rate, _entropy])
    # accs = np.array(accs).reshape(units_len, steps)
    # # S = np.array(S).reshape(units_len, steps)
    # # t_S = np.array(t_S).reshape(units_len, steps)
    # # r_S = np.array(r_S).reshape(units_len, steps)
    # errors = np.array(errors).reshape(units_len, steps)
    # dims = np.array(dims).reshape(units_len, steps)

    import csv
    with open(f'./data_out/{coords_file.split("/")[-1][:-4]}_units.csv', 'w') as f:
        writer=csv.writer(f)
        writer.writerows(units_combinations)
    #     for line in units_combinations:
    exit()


    fig_size = np.array([6.4, 4.8]) * 2  # * np.array([1, len(units_sizes) + 1])
    plt.rcParams['font.size'] = 12
    font_size = 'xx-large'
    title_size = 'xx-large'
    tick_size = 'x-large'
    legend_label = ['', 'median of ', 'average of ']
    colors = np.random.randint(0, 255, (reservoir_size, 3)) / 255

    figs = [plt.figure(figsize=fig_size) for i in range(units_len + 3)]
    axes = [figs[i].add_subplot() for i in range(units_len + 3)]
    for i in range(units_len + 3):
        figs[i].tight_layout(pad=4)

    # axes[0].boxplot([accs[i] for i in range(units_len)])
    # set_graph_params(axes[0], 'accuracy / units size', 'units', 'accuracy', title_size, font_size, tick_size)
    # today = date.today()
    # prev_image_title = f'{"{:02d}{:02d}".format(today.month, today.day)}_{coords_file.split("/")[-1][:-4]}'
    # with PdfPages(f'./image_out/{prev_image_title}_acc.pdf') as pdf:
    #     pdf.savefig(figs[0])
    # exit()

    for i in range(units_len):
        axes[0].scatter(errors[i], dims[i], color=f'C{i}', marker='x', alpha=0.5)
        # median
        axes[0].scatter(np.median(errors[i]), np.median(dims[i]), color=f'C{i}', marker='o', s=12 ** 2)
        # average
        axes[0].scatter(np.average(errors[i]), np.average(dims[i]), color=f'C{i}', marker='*', s=12 ** 2)
    axes[0].legend([f'{p}{u} units' for u in units_sizes for p in legend_label], ncol=3)
    set_graph_params(axes[0], 'correlation dimensions / error', 'error', 'dimension', title_size, font_size, tick_size)

    today = date.today()
    prev_image_title = f'{"{:02d}{:02d}".format(today.month, today.day)}_{coords_file.split("/")[-1][:-4]}'
    # post_image_title = input('save figure? [title/n]: ')
    with PdfPages(f'./image_out/{prev_image_title}_cdim.pdf') as pdf:
        pdf.savefig(figs[0])
    print('saved')
    exit()

    # for i in range(units_len):
    #     axes[0].scatter(errors[i], S[i], color=f'C{i}', marker='x', alpha=0.5)
    #     # median
    #     axes[0].scatter(np.median(errors[i]), np.median(S[i]), color=f'C{i}', marker='o', s=12 ** 2)
    #     # average
    #     axes[0].scatter(np.average(errors[i]), np.average(S[i]), color=f'C{i}', marker='*', s=12 ** 2)
    # axes[0].legend([f'{p}{u} units' for u in units_sizes for p in legend_label], loc='lower left', ncol=2)
    # set_graph_params(axes[0], 'error / entropy', 'error', 'entropy', title_size, font_size, tick_size)
    # slope, intercept, stat_info = linear_regression(errors.flatten(), S.flatten())
    # axes[0].plot(errors.flatten(), errors.flatten() * slope + intercept, linestyle='--', color=f'C9')
    # corr, corr_p_value = stats.pearsonr(errors.flatten(), S.flatten())
    #
    # for i in range(units_len):
    #     axes[1].scatter(errors[i], t_S[i], color=f'C{i}', marker='x', alpha=0.5)
    #     axes[1].scatter(np.median(errors[i]), np.median(t_S[i]), color=f'C{i}', marker='o', s=12 ** 2)
    #     axes[1].scatter(np.average(errors[i]), np.average(t_S[i]), color=f'C{i}', marker='*', s=12 ** 2)
    # axes[1].legend([f'{p}{u} units' for u in units_sizes for p in legend_label], ncol=2)
    # set_graph_params(axes[1], 'error / tsallis entropy', 'error', 'entropy', title_size, font_size, tick_size)
    # t_slope, t_intercept, t_stat_info = linear_regression(errors.flatten(), t_S.flatten())
    # axes[1].plot(errors.flatten(), errors.flatten() * t_slope + t_intercept, linestyle='--', color=f'C9')
    # t_corr, t_corr_p_value = stats.pearsonr(errors.flatten(), t_S.flatten())
    #
    # for i in range(units_len):
    #     axes[2].scatter(errors[i], r_S[i], color=f'C{i}', marker='x', alpha=0.5)
    #     axes[2].scatter(np.median(errors[i]), np.median(r_S[i]), color=f'C{i}', marker='o', s=12 ** 2)
    #     axes[2].scatter(np.average(errors[i]), np.average(r_S[i]), color=f'C{i}', marker='*', s=12 ** 2)
    # axes[2].legend([f'{p}{u} units' for u in units_sizes for p in legend_label], ncol=2)
    # set_graph_params(axes[2], 'error / renyi entropy', 'errorentropy.py', 'entropy', title_size, font_size, tick_size)
    # r_slope, r_intercept, r_stat_info = linear_regression(errors.flatten(), r_S.flatten())
    # axes[2].plot(errors.flatten(), errors.flatten() * r_slope + r_intercept, linestyle='--', color=f'C9')
    # r_corr, r_corr_p_value = stats.pearsonr(errors.flatten(), r_S.flatten())

    # for i in range(units_len):
    #     for j in range(steps):
    #         # axes[i + 3].plot(stft_freq[high_pass * 3:stft_N // 2], distributions[j, :, i][high_pass * 3:stft_N // 2],
    #         #                  linewidth=0.2, color=f'C{i}')  # , alpha=accs[i * steps + j])
    #         axes[i + 3].plot(stft_freq[high_pass * 3:low_pass * 3], distributions[j, :, i][high_pass * 3:low_pass * 3],
    #                          linewidth=0.2, color=f'C{i}')
    #     set_graph_params(axes[i + 3], 'Power Spectrum', 'frequency [Hz]', 'power', title_size, font_size, tick_size)
    # # for i in range(3, len(units_sizes) + 2):
    # #     axes[2].get_shared_y_axes().join(axes[2], axes[i])

    today = date.today()
    prev_image_title = f'{"{:02d}{:02d}".format(today.month, today.day)}_{coords_file.split("/")[-1][:-4]}'
    post_image_title = input('save figure? [title/n]: ')
    # post_image_title = f't{"{:02d}".format(int(t_param * 10))}r{"{:02d}".format(int(r_param * 10))}' \
    #                    f'_{units_sizes[0]}-{units_sizes[-1]}_cut{high_pass}-{low_pass}hz'
    for i in range(len(units_sizes) + 3):
        with PdfPages(f'./image_out/{prev_image_title}_entropy_{post_image_title}_{i}.pdf') as pdf:
            pdf.savefig(figs[i])
        print('saved')

    # with open(f'./data_out/{prev_image_title}_entropy_{post_image_title}.txt', 'w') as f:
    #     f.write(f'[entropy]\n'
    #             f'  correlation: {corr}, p: {corr_p_value}\n'
    #             f'  regression: {slope}x + {intercept}, R^2: {stat_info[0]}, p:{stat_info[1]}, std err: {stat_info[2]}\n'
    #             f'[tsallis entropy]\n'
    #             f'  correlation: {t_corr}, p: {t_corr_p_value}\n'
    #             f'  regression: {t_slope}x + {t_intercept}, R^2: {t_stat_info[0]}, p:{t_stat_info[1]}, std err: {t_stat_info[2]}\n'
    #             f'[renyi entropy]\n'
    #             f'  correlation: {r_corr}, p: {r_corr_p_value}\n'
    #             f'  regression: {r_slope}x + {r_intercept}, R^2: {r_stat_info[0]}, p:{r_stat_info[1]}, std err: {r_stat_info[2]}')

    # fig, axes = plt.subplots(nrows=len(units_sizes) + 1, figsize=fig_size)
    #
    # for i in range(len(units_sizes)):
    #     axes[0].scatter(accs[i * test_steps:(i + 1) * test_steps], es[i * test_steps:(i + 1) * test_steps],
    #                     color=f'C{i}', marker='x', alpha=0.8)
    #     # median
    #     axes[0].scatter(np.median(accs[i * test_steps:(i + 1) * test_steps]),
    #                     np.median(es[i * test_steps:(i + 1) * test_steps]), color=f'C{i}', marker='o', s=12 ** 2)
    #     # average
    #     axes[0].scatter(np.average(accs[i * test_steps:(i + 1) * test_steps]),
    #                     np.average(es[i * test_steps:(i + 1) * test_steps]), color=f'C{i}', marker='*', s=12 ** 2)
    # axes[0].legend([f'{p}{u} units' for u in units_sizes for p in ['', 'median of ', 'average of ']])
    # axes[0].set_xlabel('accuracy', fontsize=font_size)
    # axes[0].set_ylabel('entropy', fontsize=font_size)
    # axes[0].set_title('accuracy / entropy', fontsize=title_size)
    # axes[0].tick_params(labelsize=tick_size)

    # for i in range(len(units_sizes)):
    #     for j in range(test_steps):
    #         axes[i + 1].plot(stft_freq[3:stft_N // 2], distributions[j, :, i][3:stft_N // 2], linewidth=0.2,
    #                          color=f'C{i}')
    #     axes[i + 1].set_title('Power Spectrum', fontsize=title_size)
    #     axes[i + 1].set_xlabel('frequency [Hz]', fontsize=font_size)
    #     axes[i + 1].set_ylabel('power', fontsize=font_size)
    #     axes[i + 1].tick_params(labelsize=tick_size)
    #
    # slope, intercept, r_value, p_value, std_err = stats.linregress(accs, es)
    # print(f'R-squared: {r_value ** 2}, p-value: {p_value}, standard error: {std_err}')
    # axes[0].plot(accs, accs * slope + intercept, linestyle='--', color=f'C{len(units_sizes)}')

    # ax2 = fig.add_subplot(212)
    # for i in range(len(units_sizes)):
    #     for j in range(test_steps):
    #         ax2.scatter(np.arange(classified_label.size), results[j, :, i], color=f'C{i + 1}', marker='x', alpha=0.2)
    # ax2.plot(classified_label, linewidth=0.5, color='C0')
    # ax2.tick_params(labelsize=tick_size)

    # today = date.today()
    # prev_image_title = f'{"{:02d}{:02d}".format(today.month, today.day)}_{coords_file.split("/")[-1][:-4]}'
    # post_image_title = input('save figure? [title/n]: ')
    # if post_image_title != 'n':
    #     with PdfPages(f'./image_out/{prev_image_title}_{post_image_title}.pdf') as pdf:
    #         pdf.savefig(fig)
    #     print('saved')

    # with open(f'./data_out/{"{:02d}{:02d}".format(today.month, today.day)}_{std_file.split("/")[-1][:-4]}_result.txt',
    #           'w') as f:
    #     for result in records:
    #         f.write(f'{",".join([str(i) for i in result])}\n')
