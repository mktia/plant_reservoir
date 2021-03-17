from datetime import datetime
from typing import List, Set, Tuple, Union
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, stats
from tqdm import tqdm
from utils_cp import Calculate, FileController


def classify(output: cp.ndarray) -> cp.ndarray:
    return cp.argmax(output, axis=1)


def correlation_integral(distance_X: cp.ndarray, r: float) -> int:
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


def corr_integral_of_each_params(distance_X: cp.ndarray, r_params: cp.ndarray, units_size, count) -> cp.ndarray:
    Cs = cp.array([correlation_integral(distance_X, r) for r in r_params])
    # if count < 1:
    #     fig = plt.figure()
    #     ax = fig.add_subplot()
    #     ax.scatter(np.log(r_params), np.log(Cs))
    #     set_graph_params(ax, f'log C(r) / log r [{units_size} units]', 'log r', 'log C(r)', 'x-large', 'x-large',
    #                      'large')
    #     with PdfPages(f'./image_out/correlation_integral_{units_size}u') as pdf:
    #         pdf.savefig(fig)
    return Cs


def distance(X: cp.ndarray) -> cp.ndarray:
    """各点の差の絶対値による距離を求める

    Args:
        X: 時系列データ (Nステップ x m次元)

    Returns:
        各点間の距離の行列
    """
    N, _ = X.shape
    distance_X = cp.zeros((N, N - 1))
    for i in range(N):
        diff_xs = X[i, :] - cp.vstack((X[:i, :], X[i + 1:, :]))
        distance_X[i, :] = cp.sum(cp.abs(diff_xs), axis=1)
    return distance_X


def dot(X: cp.ndarray, Y: cp.ndarray) -> cp.ndarray:
    return cp.dot(X, Y)


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
        if len(units) == num_of_select_units and len(removed_units & units) == 0:
            break
    return list(units)


def linear_regression(x: cp.ndarray, y: cp.ndarray) -> Tuple[float, float, List[float]]:
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f'slope: {slope}, correlation: {r_value}, p-value: {p_value}, standard error: {std_err}')
    return slope, intercept, [r_value, p_value, std_err]


def ridge_regression(data: cp.ndarray, label: cp.ndarray) -> cp.ndarray:
    reg = 1e4
    _, eye_size = data.shape
    pre_inv = dot(data.T, data) + reg * cp.eye(eye_size)
    inv = linalg.inv(cp.asnumpy(pre_inv))
    return dot(dot(label, data), cp.asarray(inv))


def set_graph_params(ax, title: str, xlabel: str, ylabel: str, title_size: Union[str, float, int],
                     font_size: Union[str, float, int], label_size: Union[str, float, int]) -> None:
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(labelsize=label_size)


def main():
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    coords_file, label_file = './data_in/0513_l123r123_q1_md20_ws15_fixed.csv', './data_in/1111_l123r123_q1_md20_ws15_label.csv'
    coords_data = np.loadtxt(coords_file, delimiter=',')
    label_data = np.loadtxt(label_file, delimiter=',')
    print(f'coords: {coords_file.split("/")[-1]}, label: {label_file.split("/")[-1]}')

    T = 8000
    test_T = 500
    class_size = 6
    time_size, reservoir_size = coords_data.shape

    # l123r123
    coords_data = cp.vstack((coords_data[:9315], coords_data[27585:36900], coords_data[9315:18360],
                             coords_data[36900:46245], coords_data[18360:27585], coords_data[46245:]))
    # end_frames = [9315, 18360, 27585, 36900, 46245, 55830]
    # lr1lr2lr3
    end_frames = [9315, 18630, 27675, 37020, 46245, 55830]

    # data preprocessing
    X = cp.ones((1, reservoir_size))
    test_data = cp.copy(X)
    label = cp.ones((1, class_size))
    test_label = cp.copy(label)
    start = 0
    for end in end_frames:
        X = cp.vstack((X, coords_data[start:start + T]))
        test_data = cp.vstack((test_data, coords_data[start + T:start + T + test_T]))
        label = cp.vstack((label, label_data[start:start + T]))
        test_label = cp.vstack((test_label, label_data[start + T:start + T + test_T]))
        start = end
    X = X[1:]
    test_data = test_data[1:]
    label = label[1:]
    test_label = test_label[1:]

    Yt_T = label.T
    stride, window = 1, 11

    np.random.seed(32)

    # ignore units
    ignore_indices = set([i for i, x in enumerate(X[0, :]) if x == 0])
    print(f'ignore indices: {sorted(list(ignore_indices))}')

    units_sizes = [3, 5, 10, 20]
    units_len = len(units_sizes)
    steps = 100
    test_length = (test_T * class_size - window) // stride + 1
    classified = cp.zeros((steps, test_length, len(units_sizes)))
    averaged_test_label = test_label[5:-5]
    # classified_label = cp.argmax(averaged_test_label, axis=1)
    rad_params = cp.array([10 ** ((- num + 4) / 4) for num in range(12)][::-1])
    log_rad_params = cp.log(rad_params)
    accs, errors, dims = [[] for i in range(3)]
    units_combinations = []
    for k, units_size in enumerate(units_sizes):
        for count in tqdm(range(steps)):
            random_indices = get_random_units(reservoir_size, units_size, ignore_indices)
            units_combinations.append(random_indices)
            X_selected = cp.hstack([X[:, i][:, cp.newaxis] for i in random_indices])
            Wout = ridge_regression(X_selected, Yt_T)
            # length = (test_T * class_size - window) // stride + 1
            averaged_test_data = Calculate.moving_average(test_data, window, stride)
            averaged_test_data_selected = cp.hstack(
                [averaged_test_data[:, i][:, cp.newaxis] for i in random_indices])

            distance_X = distance(X_selected)
            C = corr_integral_of_each_params(distance_X, rad_params, units_size, count)
            C = cp.log(C)
            x = cp.array([log_rad_params[i + 1] - log_rad_params[i] for i in range(len(C) - 1)])
            y = cp.array([C[i + 1] - C[i] for i in range(len(C) - 1)])
            dy = y / x
            dy = cp.nan_to_num(dy)
            dy = dy[dy < 100]
            median_dy = cp.asnumpy(cp.max(dy))
            dims.append(median_dy.astype(np.float))

            Y = dot(Wout, averaged_test_data_selected.T).T
            # classification
            classified[count, :, k] = classify(Y)

            Y_softmax = Calculate.softmax(Y)
            # error
            cross_entropy = Calculate.cross_entropy(Y_softmax, averaged_test_label)
            cross_entropy = cp.asnumpy(cross_entropy)
            errors.append(cross_entropy.astype(np.float))

            # calculate accuracy rate
            # wrong_count = cp.count_nonzero(classified[count, :, k] - classified_label)
            # accuracy_rate = cp.asnumpy(1 - wrong_count / length)
            # accs.append(accuracy_rate.astype(np.float))

    # accs = np.array(accs).reshape(units_len, steps)
    errors = np.array(errors).reshape(units_len, steps)
    dims = np.array(dims).reshape(units_len, steps)
    units_combinations = np.array(units_combinations)

    dt = datetime.now()
    prev_title = f'{"{:02d}{:02d}-{:02d}{:02d}".format(dt.month, dt.day, dt.hour, dt.minute)}_{coords_file.split("/")[-1][:-4]}'

    np.savetxt(f'./data_out/{prev_title}_errors.csv', errors, delimiter=',')
    np.savetxt(f'./data_out/{prev_title}_dims.csv', dims, delimiter=',')
    np.savetxt(f'./data_out/{prev_title}_units.csv', units_combinations, delimiter=',')

    fig_size = np.array([6.4, 4.8]) * 2
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'serif'
    font_size = 'xx-large'
    title_size = 'xx-large'
    tick_size = 'x-large'
    legend_label = ['', 'median of ', 'average of ']

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot()
    fig.tight_layout(pad=4)

    for i in range(units_len):
        ax.scatter(errors[i], dims[i], color=f'C{i}', marker='x', alpha=0.5)
        # median
        ax.scatter(np.median(errors[i]), np.median(dims[i]), color=f'C{i}', marker='o', s=12 ** 2)
        # average
        ax.scatter(np.average(errors[i]), np.average(dims[i]), color=f'C{i}', marker='*', s=12 ** 2)
    ax.legend([f'{p}{u} units' for u in units_sizes for p in legend_label], ncol=2)
    set_graph_params(ax, 'correlation dimensions / error', 'error', 'dimension', title_size, font_size, tick_size)

    slope, intercept, info = linear_regression(errors.flatten(), dims.flatten())
    ax.plot(errors.flatten(), slope * errors.flatten() + intercept, color='C9')

    fig.savefig(f'./image_out/{prev_title}_corr_dim.eps', format='eps')
    print('saved')

    with open(f'./data_out/{prev_title}.txt', 'w') as f:
        f.write(f'correlation coefficient: {info[0]}')
    #
    # min_dim = np.min(dims[0])
    # min_dim_idx = np.argmin(dims[0])
    # min_dim_combination = units_combinations[min_dim_idx]
    # attractor = cp.asnumpy(cp.hstack([X[:, i][:, np.newaxis] for i in min_dim_combination]))
    #
    # fig = plt.figure(figsize=fig_size * 2)
    # ax3d = fig.add_subplot(111, projection='3d')
    # for i in range(class_size):
    #     ax3d.scatter(attractor[T * i:T * (i + 1), 0], attractor[T * i:T * (i + 1), 1], attractor[T * i:T * (i + 1), 2],
    #                  marker='.')
    # ax3d.set_title(f'attractor (dim: {min_dim})')
    # ax3d.set_xlabel(f'unit {min_dim_combination[0]}')
    # ax3d.set_ylabel(f'unit {min_dim_combination[1]}')
    # ax3d.set_zlabel(f'unit {min_dim_combination[2]}')
    # units_ids = '-'.join([str(i) for i in min_dim_combination])
    # # with PdfPages(f'./image_out/{prev_title}_attractor_{units_ids}.pdf') as pdf:
    # #     pdf.savefig(fig)
    # fig.savefig(f'./image_out/{prev_title}_attractor_{units_ids}.eps', format='eps')
    # print('saved')


if __name__ == '__main__':
    with cp.cuda.Device(1):
        main()
