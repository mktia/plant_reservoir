from datetime import datetime
import itertools
from typing import List, Set, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, stats
from sklearn.decomposition import PCA
from tqdm import tqdm
from utils import Calculate, FileController


def classify(output: np.ndarray) -> np.ndarray:
    return np.argmax(output, axis=1)


def get_random_units(size: int, num_of_select_units: int, removed_units: Set[int]) -> List[int]:
    """Randomly obtains the index of a specified number of units

    Args:
        random_size: Number of units that can be taken
        num_of_select_units: Number of units to select
        removed_units: Index of units to be removed

    Returns:
        Randomly obtained index list
    """
    units = None
    valid_indices = list(set(np.arange(size).tolist()) - set(removed_units))
    while True:
        choice = np.random.choice(valid_indices, num_of_select_units)
        if len(set(choice % (size / 2))) != num_of_select_units:
            continue
        units = set(choice)
        if len(units) == num_of_select_units:
            break
    return list(units)


def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, List[float]]:
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f'slope: {slope}, correlation: {r_value}, p-value: {p_value}, standard error: {std_err}')
    return slope, intercept, [r_value, p_value, std_err]


def ridge_regression(data: np.ndarray, label: np.ndarray) -> np.ndarray:
    reg = 1e6
    _, eye_size = data.shape
    pre_inv = np.dot(data.T, data) + reg * np.eye(eye_size)
    inv = linalg.inv(pre_inv)
    return np.dot(np.dot(label, data), inv)


def set_graph_params(ax, title: str, xlabel: str, ylabel: str, title_size: Union[str, float, int],
                     font_size: Union[str, float, int], label_size: Union[str, float, int]) -> None:
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(labelsize=label_size)


def main():
    """Time series data analysis focusing on distance between distributions
    １．vertical axis：PCA，horizontal axis：cross entropy error
    ２．vertical axis：Sum of EMD in 30 ways to choose 2 classes，horizontal axis，cross entropy error
    ３．vertical axis：Sum of distances between distributions of one class and the other five classes using the Ward method，horizontal axis：cross entropy error
    ＊

    """
    coords_file, label_file = FileController.get_file(['csv'] * 2, './data_in')
    coords_data = np.loadtxt(coords_file, delimiter=',')
    label_data = np.loadtxt(label_file, delimiter=',')
    print(f'coords: {coords_file.split("/")[-1]}, label: {label_file.split("/")[-1]}')

    T, test_T = 8000, 500
    class_size = 6
    time_size, reservoir_size = coords_data.shape

    # data preprocessing
    # l123r123 to lr1lr2lr3
    coords_data = np.vstack((coords_data[:9315], coords_data[27585:36900], coords_data[9315:18360],
                             coords_data[36900:46245], coords_data[18360:27585], coords_data[46245:]))
    end_frames = [9315, 18630, 27675, 37020, 46245, 55830]
    X = np.ones((1, reservoir_size))
    test_data = np.copy(X)
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

    Yt_T = label.T
    stride, window = 1, 11

    np.random.seed(32)

    # ignore units
    ignore_indices = set([i for i, x in enumerate(X[0, :]) if x == 0])
    print(f'ignore indices: {sorted(list(ignore_indices))}')

    # List of number of units
    units_sizes = [5, 10, 15, 20]
    units_len = len(units_sizes)
    # Number of attempts
    steps = 100
    test_length = (test_T * class_size - window) // stride + 1
    classified = np.zeros((steps, test_length, len(units_sizes)))
    averaged_test_label = test_label[5:-5]
    errors, pca_dims, emd_list, d_list = [] * 4
    units_combinations = []
    for k, units_size in enumerate(units_sizes):
        for count in tqdm(range(steps)):
            random_indices = get_random_units(reservoir_size, units_size, ignore_indices)
            units_combinations.append(random_indices)
            X_selected = np.hstack([X[:, i][:, np.newaxis] for i in random_indices])
            Wout = ridge_regression(X_selected, Yt_T)
            length = (test_T * class_size - window) // stride + 1
            averaged_test_data = Calculate.moving_average(test_data, window, stride)
            averaged_test_data_selected = np.hstack(
                [averaged_test_data[:, i][:, np.newaxis] for i in random_indices])

            Y = np.dot(Wout, averaged_test_data_selected.T).T
            # classification
            classified[count, :, k] = classify(Y)

            Y_softmax = Calculate.softmax(Y)
            # error
            cross_entropy = Calculate.cross_entropy(Y_softmax, averaged_test_label)
            errors.append(cross_entropy)

            ## Number of axes with a cumulative PCA contribution of 90% or more (comment out when used)
            # PCA ->
            # pca = PCA(n_components=0.9, whiten=False)
            # pca.fit(X_selected)
            # ratio = np.cumsum(pca.explained_variance_ratio_)
            # pca_dim = ratio.size
            # pca_dims.append(pca_dim)
            # <- PCA

            ## Calculate EMD between all classes in 30 different ways of choosing 2 classes (comment out when used)
            # Enumerate how to choose a class
            # EMD ->
            # class_combi = itertools.combinations([i for i in range(class_size)], 2)
            # emd = 0
            # # Sum of EMD for all combinations
            # for combi in class_combi:
            #     c0, c1 = np.array(combi)*8000
            #     each_emd = Calculate.emd(X_selected[c0:c0 + 1000, :], X_selected[c1:c1 + 1000, :])
            #     emd += each_emd
            # emd_list.append(emd)
            # <- EMD

            ## Sum of distances between two distributions using Ward's method
            # Ward ->
            d = 0
            for i in range(class_size):
                target_data = X_selected[i * 8000:(i + 1) * 8000]
                other_data = np.ones(units_size)
                for j in range(class_size):
                    if i == j:
                        continue
                    other_data = np.vstack((other_data, X_selected[j * 8000:(j + 1) * 8000]))
                other_data = other_data[1:]
                d += Calculate.ward(target_data, other_data)
            d_list.append(d)
            # <- Ward

    errors = np.array(errors).reshape(units_len, steps)
    pca_dims = np.array(pca_dims).reshape(units_len, steps)
    emd_list = np.array(emd_list).reshape(units_len, steps)
    d_list = np.array(d_list).reshape(units_len, steps)

    # File output (commented out when used)
    # FileController.export_data(np.vstack((errors, pca_dims)), coords_file, title='_pca')
    # FileController.export_data(np.vstack((errors, emd_list)), coords_file, title='_emd')
    FileController.export_data(np.vstack((errors, d_list)), coords_file, title='_ward')

    # Graph output
    # Two comment-outs need to be changed according to the value you want to take on the vertical axis.
    fig_size = [12, 8]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    font_size = 24
    title_size = 30
    tick_size = 20

    fig = plt.figure(figsize=fig_size, constrained_layout=True)
    ax = fig.add_subplot()

    for i in range(units_len):
        # Comment out according to usage (1)
        # ax.scatter(errors[i], pca_dims[i], color=f'C{i}', marker='x')
        # ax.scatter(errors[i], emd_list[i], color=f'C{i}', marker='x')
        ax.scatter(errors[i], d_list[i], color=f'C{i}', marker='x')
    ax.legend([f'{u} units' for u in units_sizes], ncol=5, bbox_to_anchor=[0, 1, 1, 0.15], frameon=False, fontsize=20,
              markerscale=2)
    set_graph_params(ax, '', 'Error', 'Sum of distance', title_size, font_size, tick_size)

    # Comment out according to usage (2)
    # slope, intercept, info = linear_regression(errors.flatten(), pca_dims.flatten())
    # slope, intercept, info = linear_regression(errors.flatten(), emd_list.flatten())
    slope, intercept, info = linear_regression(errors.flatten(), d_list.flatten())
    ax.plot(errors.flatten(), slope * errors.flatten() + intercept, color='C9')
    print(f'correlation dimension: {info[0]}')

    dt = datetime.today().isoformat().replace(':', '-').split('.')[0]
    prev_image_title = f'{dt}_{coords_file.split("/")[-1][:-4]}'
    # 'tag' should be modified depending on what is saved.
    tag = 'ward'
    fig.savefig(f'./image_out/{prev_image_title}_{tag}.eps', format='eps', dpi=300)
    print('saved')


if __name__ == '__main__':
    main()
