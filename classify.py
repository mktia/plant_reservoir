from copy import deepcopy
import csv
from datetime import datetime
from itertools import combinations
from random import sample, seed
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm, trange
from utils import Calculate, FileController, GraphViewer, MatplotlibExtension


### preprocessing

def get_random_indices(valid_units: List, size: int, whole_size: int, is_duplicated=False) -> List:
    """Randomly obtains the index of a specified number of units

    Args:
        valid_units: Index list of units on the plant
        size: Number of units to select
        whole_size: Number of all units detected
        is_duplicated: Allow overlap in feature points to which the unit belongs?

    Returns:
        Randomly retrieved index list
    """
    if not is_duplicated:
        while True:
            choice = np.array(sample(set(valid_units), size))
            if len(set(choice % (whole_size / 2))) == size:
                return choice.tolist()
    return sample(set(valid_units), size)


def import_data(is_6class: bool = True):
    """Read data

    Args:
        classified_class: Classification type ('6c': 6-class classification, '24c': 24-class classification)
    """
    print('Select data and label files')
    file_types = ['csv' for i in range(2)]
    init_dir = './data_in'
    coords_file, label_file = FileController.get_file(file_types, init_dir)
    coords_data = np.loadtxt(coords_file, delimiter=',')
    label_data = np.loadtxt(label_file, delimiter=',')
    print(f'data: {coords_file}')

    return coords_file, preprocessing(coords_data, label_data, is_6class)


def preprocessing_before(data: np.ndarray, is_6class: bool):
    """Preliminary parameter setting

    Args:
        data: time series data
        classified_class: Classification Type
    """
    # Time series length of training and test data per class
    T, test_T = (8000, 500) if is_6class else (16000, 1000)
    # If the concatenation order is l123r123 video, re-connect to r1lr2lr3
    coords_data = np.vstack((data[:9315], data[27585:36900], data[9315:18360],
                             data[36900:46245], data[18360:27585], data[46245:])) if is_6class else data
    # Record video length for each class to ensure uniform time series length
    end_frames = [9315, 18630, 27675, 37020, 46245, 55830] if is_6class \
        else [60 * 60 * 5 * (i + 1) - 1 for i in range(24)]
    return T, test_T, coords_data, end_frames


def preprocessing(coords_data: np.ndarray, label_data: np.ndarray, is_6class: bool):
    """Perform data preprocessing

    Args:
        coords_data: time series data
        label_data: label data
        classified_class: Classification Type
    """
    T, test_T, data, end_frames = preprocessing_before(coords_data, is_6class)
    _, reservoir_size = data.shape
    _, class_size = label_data.shape

    # Adjust the number of frames in each class to 'T'
    start = 0
    X = np.ones((1, reservoir_size))
    test_data = np.copy(X)
    label = np.ones((1, class_size))
    test_label = np.copy(label)
    for end in end_frames:
        X = np.vstack((X, data[start:start + T, :]))
        test_data = np.vstack((test_data, data[start + T:start + T + test_T, :]))
        label = np.vstack((label, label_data[start:start + T, :]))
        test_label = np.vstack((test_label, label_data[start + T:start + T + test_T, :]))
        start = end
    X = X[1:]
    test_data = test_data[1:]
    label = label[1:]
    test_label = test_label[1:]
    test_length, _ = test_data.shape

    return reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length


### basis

def regression_6class():
    """6-class classification by ridge regression

    """
    # import data
    coords_file, (reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length) = import_data()

    # If only use the arbitrary units to train Wout, input 'y'
    if input('Select the arbitrary units? [y/n]') == 'y':
        X_tmp = np.zeros_like(X)
        # Space-separated input
        selected_units = [int(i) for i in input('units: ').split()]
        for i in selected_units:
            X_tmp[:, i] = X[:, i]
        X = X_tmp

    # Ridge Regression (reg_param is 1e6 by default)
    Wout = Calculate.ridge_regression(X, label)
    # uncomment to save Wout as csv file
    # FileController.export_data(Wout, coords_file, title='_wout')

    # moving averate of test data
    stride, window = 1, 11
    averaged_test_data = Calculate.moving_average(test_data, window, stride)
    averaged_test_label = test_label[5:-5]

    # calculate output with Wout
    Y = np.dot(Wout, averaged_test_data.T).T
    classified = np.argmax(Y, axis=1)
    wrong_count = np.count_nonzero(classified - np.hstack([np.ones(test_T) * i for i in range(class_size)])[5:-5])
    accuracy_rate = 1 - wrong_count / (test_T * class_size)
    print(f'accuracy rate: {accuracy_rate}')

    # draw the graphs
    figure_size = np.array([12, 8])
    font_size = 28
    legend_size = 22
    tick_size = 22
    anchor_options_p1 = [0, 1, 1, 0.08]
    anchor_options_p2 = [0, 1, 1, 0.15]
    plt.rcParams['font.size'] = font_size
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['font.sans-serif'] = ['Arial']

    figs = [plt.figure(figsize=figure_size) for _ in range(4)]
    axes = [figs[i].add_subplot() for i in range(4)]

    # figure of wout
    for i in range(class_size):
        axes[0].bar(np.arange(reservoir_size), Wout[i])
    axes[0].set_xlabel('Index of units', fontsize=font_size)
    axes[0].set_ylabel('Weight', fontsize=font_size)
    axes[0].legend([f'class {i}' for i in range(class_size)], ncol=3, fontsize=legend_size,
                   bbox_to_anchor=anchor_options_p2, borderaxespad=0, frameon=False)
    axes[0].tick_params(labelsize=tick_size)
    axes[0].ticklabel_format(scilimits=[-2, 6])

    # figure of output
    for i in range(class_size):
        axes[1].plot(Y[:, i], linewidth=1)
    axes[1].set_xlabel('Step', fontsize=font_size)
    axes[1].set_ylabel('Value', fontsize=font_size)
    axes[1].legend([f'C{i}' for i in range(class_size)], ncol=3, fontsize=legend_size,
                   bbox_to_anchor=anchor_options_p2, borderaxespad=0, frameon=False)
    axes[1].tick_params(labelsize=tick_size)

    # figure of averaged time series data
    axes[2].plot(averaged_test_data[:, :60], linewidth=0.8)
    axes[2].set_xlabel('Step', fontsize=font_size)
    axes[2].set_ylabel('Coordinate', fontsize=font_size)
    axes[2].tick_params(labelsize=tick_size)

    # figure of target and classified label
    axes[3].plot(np.argmax(averaged_test_label, axis=1), 'g', linewidth=0.5)
    axes[3].scatter(np.arange(Y.shape[0]), classified, marker='x', s=12, linewidth=0.5)
    axes[3].set_xlabel('Step', fontsize=font_size)
    axes[3].set_ylabel('Class', fontsize=font_size)
    axes[3].legend(['target', 'classified'], ncol=2, fontsize=legend_size, bbox_to_anchor=anchor_options_p1,
                   borderaxespad=0, frameon=False)
    axes[3].tick_params(labelsize=tick_size)

    dt = datetime.today().isoformat().replace(':', '-').split('.')[0]
    prev_image_title = f'{dt}_{coords_file.split("/")[-1][:-4]}'
    for i in range(4):
        figs[i].savefig(f'./image_out/{prev_image_title}_reg_{i}.eps', format='eps', dpi=300)


def regression_24class():
    """24 Class Classification by Ridge Regression

    """
    coords_file, (reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length) \
        = import_data(is_6class=False)

    if input('Select the arbitrary units? [y/n]') == 'y':
        X_tmp = np.zeros_like(X)
        # Space-separated input
        selected_units = [int(i) for i in input('units: ').split()]
        for i in selected_units:
            X_tmp[:, i] = X[:, i]
        X = X_tmp

    # Ridge Regression (reg_param is 1e6 by default)
    Wout = Calculate.ridge_regression(X, label)
    # uncomment to save Wout as csv file
    # FileController.export_data(Wout, coords_file, title='_wout')

    stride, window = 1, 11
    averaged_test_data = Calculate.moving_average(test_data, window, stride)
    averaged_test_label = test_label[5:-5]

    Y = np.dot(Wout, averaged_test_data.T).T
    classified = np.argmax(Y, axis=1)
    wrong_count = np.count_nonzero(classified - np.hstack([np.ones(test_T) * i for i in range(class_size)])[5:-5])
    accuracy_rate = 1 - wrong_count / (test_T * class_size)
    print(f'accuracy rate: {accuracy_rate}')

    figure_size = np.array([12, 8])
    font_size = 28
    legend_size = 18
    tick_size = 22
    anchor_options_p1 = [0, 1, 1, 0.08]
    anchor_options_p2 = [0.15, 1, 0.85, 0.22]
    plt.rcParams['font.size'] = font_size
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['font.sans-serif'] = ['Arial']
    # custom color map for 24 class
    class_colors = MatplotlibExtension.get_custom_cmap()

    figs = [plt.figure(figsize=figure_size) for _ in range(4)]
    axes = [figs[i].add_subplot() for i in range(4)]

    # figure of wout
    for i in range(class_size):
        axes[0].bar(np.arange(reservoir_size), Wout[i], color=class_colors(i))
    axes[0].set_xlabel('Index of units', fontsize=font_size)
    axes[0].set_ylabel('Weight', fontsize=font_size)
    axes[0].legend([f'C{i}' for i in range(class_size)], ncol=6, fontsize=legend_size - 2,
                   bbox_to_anchor=anchor_options_p2, borderaxespad=0, frameon=False)
    axes[0].tick_params(labelsize=tick_size)
    axes[0].ticklabel_format(scilimits=[-2, 6])

    # figure of output
    for i in range(class_size):
        axes[1].plot(Y[:, i], linewidth=1, color=class_colors(i))
    axes[1].set_xlabel('Step', fontsize=font_size)
    axes[1].set_ylabel('Value', fontsize=font_size)
    axes[1].legend([f'C{i}' for i in range(class_size)], ncol=6, fontsize=legend_size - 2,
                   bbox_to_anchor=anchor_options_p2, borderaxespad=0, frameon=False)
    axes[1].tick_params(labelsize=tick_size)

    # figure of time series data
    axes[2].plot(averaged_test_data[:, :60], linewidth=0.8)
    axes[2].set_xlabel('Step', fontsize=font_size)
    axes[2].set_ylabel('Coordinate', fontsize=font_size)
    axes[2].tick_params(labelsize=tick_size)

    # figure of target and classified label
    axes[3].plot(np.argmax(averaged_test_label, axis=1), 'g', linewidth=0.5)
    axes[3].scatter(np.arange(Y.shape[0]), classified, marker='x', s=12, linewidth=0.5)
    axes[3].set_xlabel('Step', fontsize=font_size)
    axes[3].set_ylabel('Class', fontsize=font_size)
    axes[3].legend(['target', 'classified'], ncol=2, fontsize=legend_size, bbox_to_anchor=anchor_options_p1,
                   borderaxespad=0, frameon=False)
    axes[3].tick_params(labelsize=tick_size)

    dt = datetime.today().isoformat().replace(':', '-').split('.')[0]
    prev_image_title = f'{dt}_{coords_file.split("/")[-1][:-4]}'
    for i in range(4):
        figs[i].savefig(f'./image_out/{prev_image_title}_reg_{i}.eps', format='eps', dpi=300)


### analysis

def relation_between_unit_size_and_accuracy():
    """Relationship between the number of units and the percentage of correct answers when the number of units is changed and re-studied

    """
    if input('show the graph from a csv file [y/n]: ') == 'y':
        file_name = FileController.get_file(['csv'], './data_out')[0]
        with open(file_name) as f:
            reader = csv.reader(f, delimiter=',')
            data = [float(row[0]) for row in reader]
        accs = [np.array([j for j in data[i * 1000:(i + 1) * 1000]]) for i in range(10)]
        graph = GraphViewer(fig_size=[12, 8], font_size=26, tick_size=22)
        graph.boxplot(accs, (np.arange(10) + 1).tolist(), f'{file_name.split("/")[-1][:-4]}_acc', '', 'Number of units',
                      'Accuracy rate')
        exit()

    coords_file, (reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length) = import_data()

    # ignore units
    ignore_indices = [i for i, x in enumerate(X[0, :]) if x == 0]
    print(f'ignore indices: {ignore_indices}')
    valid_indices = list(set(np.arange(reservoir_size).tolist()) - set(ignore_indices))

    has_save_file = True if input('Save as csv? [y/n]: ') == 'y' else False
    is_visible_training_accuracy = False if input('Show the accuracy of training? [y/n]: ') == 'y' else True

    np.random.seed(32)
    seed(32)
    stride, window = 1, 11
    time_step = 1000
    units_sizes = (np.arange(10) + 1).tolist()
    # units_sizes = np.arange(5, 105, 5).tolist()

    accs = [[] for i in units_sizes]
    if is_visible_training_accuracy:
        accs_train = [[] for i in units_sizes]
    results = []
    for idx in range(len(units_sizes)):
        for count in trange(time_step):
            random_indices = get_random_indices(valid_indices, units_sizes[idx], reservoir_size, is_duplicated=False)

            X_selected = np.hstack([X[:, i][:, np.newaxis] for i in random_indices])
            Wout = Calculate.ridge_regression(X_selected, label)

            length = (test_T * class_size - window) // stride + 1
            averaged_test_data = Calculate.moving_average(test_data, window, stride)
            averaged_test_data_selected = np.hstack([averaged_test_data[:, i][:, np.newaxis] for i in random_indices])
            averaged_test_label = test_label[5:-5]

            Y = np.dot(Wout, averaged_test_data_selected.T).T
            classified = np.argmax(Y, axis=1)
            classified_label = np.argmax(averaged_test_label, axis=1)
            # Accuracy rate
            wrong_count = np.count_nonzero(classified - classified_label)
            accuracy_rate = 1 - wrong_count / length
            accs[idx].append(accuracy_rate)

            if is_visible_training_accuracy:
                train_data = np.vstack([X_selected[(i + 1) * 8000 - test_T:(i + 1) * 8000] for i in range(class_size)])
                averaged_train_data = Calculate.moving_average(train_data, window, stride)
                Y_train = np.dot(Wout, averaged_train_data.T).T
                classified_train = np.argmax(Y_train, axis=1)
                wrong_count_train = np.count_nonzero(classified_train - classified_label)
                accuracy_rate_train = 1 - wrong_count_train / length
                accs_train[idx].append(accuracy_rate_train)

            if has_save_file:
                result = [accuracy_rate] + sorted(random_indices)
                results.append(result)
        accs[idx] = np.array(accs[idx])
        if is_visible_training_accuracy:
            accs_train[idx] = np.array(accs_train[idx])
    if has_save_file:
        date_tag = datetime.today().isoformat().replace(":", "-").split(".")[0]
        with open(f'./data_out/{date_tag}_{coords_file.split("/")[-1][:-4]}_acc.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(results)

    graph = GraphViewer(fig_size=[12, 8], font_size=26, tick_size=22)
    graph.boxplot(accs, (np.arange(10) + 1).tolist(), f'{coords_file.split("/")[-1][:-4]}_acc_{time_step}pattern',
                  '', 'Number of units', 'Accuracy rate')

    # plot line with error area
    if is_visible_training_accuracy:
        accs = np.array(accs)
        accs_train = np.array(accs_train)
        mean = np.array([np.mean(data, axis=1) for data in [accs, accs_train]])
        stderr = np.array([np.std(data, axis=1, ddof=1) for data in [accs, accs_train]])
        graph = GraphViewer(fig_size=[12, 8])
        graph.line_with_error(mean, units_sizes, stderr,
                              f'{coords_file.split("/")[-1][:-4]}_acc_witherr_{time_step}pattern', 'accuracy',
                              'Number of units', 'Accuracy rate', legends=['test', 'train'], line_width=1.2)


def relation_between_unit_size_and_accuracy_by_logistic():
    """Relationship between number of units and percentage of correct answers when trained using multinomial logistic regression.

    """
    if input('show the graph from a csv file [y/n]: ') == 'y':
        file_name = FileController.get_file(['csv'], './data_out')[0]
        with open(file_name) as f:
            reader = csv.reader(f, delimiter=',')
            data = [row[0] for row in reader]
        accs = [np.array([float(j) for j in data[i * 100:(i + 1) * 100]]) for i in range(10)]
        graph = GraphViewer(fig_size=[12, 8], font_size=26, tick_size=22)
        graph.boxplot(accs, (np.arange(10) + 1).tolist(), f'{file_name.split("/")[-1][:-4]}_acc_logistic', '',
                      'Number of units', 'Accuracy rate')
        exit()

    coords_file, (
        reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length) = import_data()

    ignore_indices = [i for i, x in enumerate(X[0, :]) if x == 0]
    valid_indices = list(set(np.arange(reservoir_size).tolist()) - set(ignore_indices))

    has_save_file = True if input('Save as csv?: ') == 'y' else False

    np.random.seed(32)
    seed(32)
    stride, window = 1, 11
    time_step = 100
    units_sizes = (np.arange(10) + 1).tolist()
    accs = [[] for i in units_sizes]
    results = []

    averaged_test_label = np.array([i for i in range(class_size) for j in range(test_T)])[5:-5]

    for idx, units_size in enumerate(units_sizes):
        for count in trange(time_step):
            random_indices = get_random_indices(valid_indices, units_size, reservoir_size)

            X_selected = np.hstack([X[:, i][:, np.newaxis] for i in random_indices])
            label = np.hstack(
                (np.zeros(T), np.ones(T), np.ones(T) * 2, np.ones(T) * 3, np.ones(T) * 4, np.ones(T) * 5))
            averaged_test_data = Calculate.moving_average(test_data, window, stride)
            averaged_test_data_selected = np.hstack(
                [averaged_test_data[:, i][:, np.newaxis] for i in random_indices])

            model = LogisticRegression(max_iter=1500).fit(X_selected, label)
            # predict=model.predict(averaged_test_data_selected)
            accuracy_rate = model.score(averaged_test_data_selected, averaged_test_label)
            accs[idx].append(accuracy_rate)

            if has_save_file:
                result = [accuracy_rate] + sorted(random_indices)
                results.append(result)
        accs[idx] = np.array(accs[idx])

    if has_save_file:
        date_tag = datetime.today().isoformat().replace(":", "-").split(".")[0]
        with open(f'./data_out/{date_tag}_{coords_file.split("/")[-1][:-4]}_acc_logistic.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(results)

    graph = GraphViewer(fig_size=[12, 8], font_size=24, tick_size=22)
    graph.boxplot(accs, (np.arange(10) + 1).tolist(), f'{coords_file.split("/")[-1][:-4]}_acc_logistic', '',
                  'Number of units', 'Accuracy rate')


def relation_between_fp_amplitude_variance_and_error():
    """Relationship between amplitude of feature points and cross-entropy error

    """
    coords_file, (
        reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length) = import_data()

    # ignore units
    ignore_indices = [i for i, x in enumerate(X[0, :]) if x == 0]
    valid_indices = list(set(np.arange(reservoir_size).tolist()) - set(ignore_indices))

    np.random.seed(32)
    seed(32)
    stride, window = 1, 11

    variances = np.zeros((class_size, reservoir_size))
    for i in range(class_size):
        tmp = np.var(X[i * 8000:(i + 1) * 8000], axis=0)
        # variances[i, :] = (tmp > sorted(tmp)[-100]) * np.ones_like(tmp)
        variances[i, :] = (tmp > np.median(tmp)) * np.ones_like(tmp)
    each_sum_var = np.sum(variances, axis=0)

    time_steps = 100
    units_sizes = (np.arange(10) + 1) * 2
    # units_sizes = [10]
    errors, var_scores = [], []
    for idx, units_size in enumerate(units_sizes):
        for count in tqdm(range(time_steps)):
            # pick up some units
            random_indices = get_random_indices(valid_indices, units_size, reservoir_size)

            X_selected = np.hstack([X[:, i][:, np.newaxis] for i in random_indices])
            Wout = Calculate.ridge_regression(X_selected, label)

            length = (test_T * class_size - window) // stride + 1
            averaged_test_data = Calculate.moving_average(test_data, window, stride)
            averaged_test_data_selected = np.hstack(
                [averaged_test_data[:, i][:, np.newaxis] for i in random_indices])
            averaged_test_label = test_label[5:-5]

            Y = np.dot(Wout, averaged_test_data_selected.T).T
            Y_softmax = Calculate.softmax(Y)
            # error
            cross_entropy = Calculate.cross_entropy(Y_softmax, averaged_test_label)
            errors.append(cross_entropy)

            indices = np.zeros(reservoir_size)
            for random_idx in random_indices:
                indices += (np.arange(reservoir_size) == random_idx) * np.ones(reservoir_size)
            # 2 >
            var_score = each_sum_var * indices
            var_score = var_score[var_score > 0]
            # var_scores.append(np.var(var_score))
            # 3
            var_scores.append(np.mean(var_score))
            # 4
            # var_scores.append(np.max(var_score))
            # < 2
            # sum(var_score), max or mean
            # var_scores.append(np.sum(var_score))
    errors = np.array(errors).reshape((len(units_sizes), time_steps))
    var_scores = np.array(var_scores).reshape((len(units_sizes), time_steps))

    slope, intercept, info = Calculate.linear_regression(errors.flatten(), var_scores.flatten())

    legend = [f'{i} units' for i in units_sizes]

    graph = GraphViewer(fig_size=[12, 8])
    graph.scatter(errors, var_scores, f'{coords_file.split("/")[-1][:-4]}_var_score_mean_median',
                  xlabel='cross entropy',
                  ylabel='var_score', slope=slope, intercept=intercept, legend=legend)


def accuracy_of_classify():
    """Output the percentage of correct classifications

    """
    coords_file, (
        reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length) = import_data()

    stride, window = 1, 11
    length = (test_T * class_size - window) // stride + 1

    Wout = Calculate.ridge_regression(X, label)

    averaged_test_data = Calculate.moving_average(test_data, window, stride)
    averaged_test_label = test_label[5:-5]

    Y = np.dot(Wout, averaged_test_data.T).T
    classified = np.argmax(Y, axis=1)
    classified_label = np.argmax(averaged_test_label, axis=1)

    # Accuracy rate
    wrong_count = np.count_nonzero(classified - classified_label)
    accuracy_rate = 1 - wrong_count / length

    print(f'Accuracy rate: {accuracy_rate}')


def delay():
    """Introduction of time delay

    """
    coords_file, (reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length) = import_data()

    ignore_indices = [i for i, x in enumerate(X[0, :]) if x == 0]
    valid_indices = list(set(np.arange(reservoir_size).tolist()) - set(ignore_indices))

    np.random.seed(32)
    seed(32)
    stride, window = 1, 11
    time_step = 1000
    units_size = 4
    d, tau_unit = 3, 3
    tau_size = ((np.arange(0, 20) + 1) * tau_unit * d).tolist()
    accs = [[] for i in tau_size]
    unit_idxs = [get_random_indices(valid_indices, units_size, reservoir_size) for i in range(time_step)]

    length = (test_T * class_size - window) // stride + 1

    new_T = 7800
    new_test_T = 300

    tmp_acc = []
    for count in range(time_step):
        random_indices = unit_idxs[count]

        X_selected = np.hstack([X[:, i][:, np.newaxis] for i in random_indices])
        X_custom = np.vstack([X_selected[i * T:i * T + new_T] for i in range(class_size)])
        label_custom = np.vstack([label[i * T:i * T + new_T] for i in range(class_size)])

        Wout = Calculate.ridge_regression(X_custom, label_custom)
        test_data_selected = np.hstack([test_data[:, i][:, np.newaxis] for i in random_indices])
        test_data_custom = np.vstack(
            [test_data_selected[i * test_T:i * test_T + new_test_T] for i in range(class_size)])
        averaged_test_data = Calculate.moving_average(test_data_custom, window, stride)
        averaged_test_label = np.vstack([test_label[i * test_T:i * test_T + new_test_T] for i in range(class_size)])[
                              5:-5]

        Y = np.dot(Wout, averaged_test_data.T).T
        classified = np.argmax(Y, axis=1)
        classified_label = np.argmax(averaged_test_label, axis=1)
        # Accuracy rate
        wrong_count = np.count_nonzero(classified - classified_label)
        accuracy_rate = 1 - wrong_count / length
        tmp_acc.append(accuracy_rate)
    tmp_acc = np.array(tmp_acc)

    for idx in trange(len(tau_size)):
        tau = tau_size[idx]
        for count in range(time_step):
            random_indices = unit_idxs[count]

            X_selected = np.hstack([X[:, i][:, np.newaxis] for i in random_indices])
            X_delay = np.vstack([np.hstack(
                (X_selected[i * T:i * T + new_T], X_selected[i * T + tau:i * T + new_T + tau]))
                for i in range(class_size)])
            label_delay = np.vstack([label[i * T:i * T + new_T] for i in range(class_size)])

            Wout = Calculate.ridge_regression(X_delay, label_delay)
            test_data_selected = np.hstack([test_data[:, i][:, np.newaxis] for i in random_indices])
            test_data_delay = np.vstack(
                [np.hstack((test_data_selected[i * test_T:i * test_T + new_test_T],
                            test_data_selected[i * test_T + tau:i * test_T + new_test_T + tau])) for i in
                 range(class_size)])
            averaged_test_data = Calculate.moving_average(test_data_delay, window, stride)
            averaged_test_label = np.vstack(
                [test_label[i * test_T:i * test_T + new_test_T] for i in range(class_size)])[5:-5]

            Y = np.dot(Wout, averaged_test_data.T).T
            classified = np.argmax(Y, axis=1)
            classified_label = np.argmax(averaged_test_label, axis=1)
            # Accuracy rate
            wrong_count = np.count_nonzero(classified - classified_label)
            accuracy_rate = 1 - wrong_count / length
            accs[idx].append(accuracy_rate)
        accs[idx] = np.array(accs[idx])
    accs.insert(0, tmp_acc)

    graph = GraphViewer(fig_size=[12, 8])
    graph.boxplot(accs, [0] + tau_size, f'{coords_file.split("/")[-1][:-4]}_acc_delay_d{d}_tau{tau_unit}',
                  'accuracy', 'step', 'Accuracy rate')


def delay_expansion():
    if input('show the graph from a csv file [y/n]: ') == 'y':
        file_name = FileController.get_file(['csv'], './data_out')[0]
        d_size, units_size = 20, 3
        data = np.loadtxt(file_name, delimiter=',')[:, units_size:]
        accs = [data[:, i] for i in range(d_size + 1)]
        graph = GraphViewer(fig_size=[12, 8], font_size=26, tick_size=22)
        graph.boxplot(accs, ((np.arange(d_size + 1) + 1) * units_size).tolist(),
                      f'{file_name.split("/")[-1][:-4]}', '', 'Number of units', 'Accuracy rate')
        exit()

    coords_file, (reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length) = import_data()

    ignore_indices = [i for i, x in enumerate(X[0, :]) if x == 0]
    valid_indices = list(set(np.arange(reservoir_size).tolist()) - set(ignore_indices))

    np.random.seed(32)
    seed(32)
    stride, window = 1, 11
    time_step = 1000
    units_size = 4
    d_size = 20
    tau_unit = 5
    accs = [[] for i in range(d_size)]
    errs = [[] for i in range(d_size)]
    unit_idxs = [get_random_indices(valid_indices, units_size, reservoir_size) for i in range(time_step)]

    new_T = 7800
    new_test_T = 300
    length = (new_test_T * class_size - window) // stride + 1

    tmp_acc, tmp_err = [], []
    for count in range(time_step):
        random_indices = unit_idxs[count]

        X_selected = np.hstack([X[:, i][:, np.newaxis] for i in random_indices])
        X_custom = np.vstack([X_selected[i * T:i * T + new_T] for i in range(class_size)])
        label_custom = np.vstack([label[i * T:i * T + new_T] for i in range(class_size)])

        Wout = Calculate.ridge_regression(X_custom, label_custom)
        test_data_selected = np.hstack([test_data[:, i][:, np.newaxis] for i in random_indices])
        test_data_custom = np.vstack(
            [test_data_selected[i * test_T:i * test_T + new_test_T] for i in range(class_size)])
        averaged_test_data = Calculate.moving_average(test_data_custom, window, stride)
        averaged_test_label = np.vstack([test_label[i * test_T:i * test_T + new_test_T] for i in range(class_size)])[
                              5:-5]

        Y = np.dot(Wout, averaged_test_data.T).T
        classified = np.argmax(Y, axis=1)
        classified_label = np.argmax(averaged_test_label, axis=1)
        # Accuracy rate
        wrong_count = np.count_nonzero(classified - classified_label)
        accuracy_rate = 1 - wrong_count / length
        tmp_acc.append(accuracy_rate)
        # error
        Y_softmax = Calculate.softmax(Y)
        cross_entropy = Calculate.cross_entropy(Y_softmax, averaged_test_label)
        tmp_err.append(cross_entropy)
    tmp_acc = np.array(tmp_acc)
    tmp_err = np.array(tmp_err)

    for idx in trange(d_size):
        tau_list = np.arange(0, idx + 2) * tau_unit
        for count in range(time_step):
            random_indices = unit_idxs[count]

            X_selected = np.hstack([X[:, i][:, np.newaxis] for i in random_indices])
            X_delay = np.vstack([np.hstack([X_selected[i * T + tau:i * T + new_T + tau] for tau in tau_list]) for i in
                                 range(class_size)])
            label_delay = np.vstack([label[i * T:i * T + new_T] for i in range(class_size)])

            Wout = Calculate.ridge_regression(X_delay, label_delay)
            test_data_selected = np.hstack([test_data[:, i][:, np.newaxis] for i in random_indices])
            test_data_delay = np.vstack(
                [np.hstack([test_data_selected[i * test_T + tau:i * test_T + new_test_T + tau] for tau in tau_list]) for
                 i in range(class_size)])
            averaged_test_data = Calculate.moving_average(test_data_delay, window, stride)
            averaged_test_label = np.vstack(
                [test_label[i * test_T:i * test_T + new_test_T] for i in range(class_size)])[5:-5]

            Y = np.dot(Wout, averaged_test_data.T).T
            classified = np.argmax(Y, axis=1)
            classified_label = np.argmax(averaged_test_label, axis=1)
            # Accuracy rate
            wrong_count = np.count_nonzero(classified - classified_label)
            accuracy_rate = 1 - wrong_count / length
            accs[idx].append(accuracy_rate)
            # error
            Y_softmax = Calculate.softmax(Y)
            cross_entropy = Calculate.cross_entropy(Y_softmax, averaged_test_label)
            errs[idx].append(cross_entropy)
        accs[idx] = np.array(accs[idx])
        errs[idx] = np.array(errs[idx])
    accs.insert(0, tmp_acc)
    errs.insert(0, tmp_err)

    # export graph
    graph = GraphViewer(fig_size=[12, 8])
    graph.boxplot(accs, (np.arange(d_size + 1) * units_size).tolist(),
                  f'{coords_file.split("/")[-1][:-4]}_acc_ext{d_size}_delay_tau{tau_unit}',
                  r'$\tau=5$', 'Number of units', 'Accuracy rate')
    graph = GraphViewer(fig_size=[12, 8])
    graph.boxplot(errs, (np.arange(d_size + 1) * units_size).tolist(),
                  f'{coords_file.split("/")[-1][:-4]}_err_ext{d_size}_delay_tau{tau_unit}',
                  r'$\tau=5$', 'Number of units', 'Cross entropy error')

    # export data
    unit_idxs = np.array(unit_idxs)
    accs = np.array(accs).T
    res = np.hstack((unit_idxs, accs))
    FileController.export_data(res, coords_file, title=f'acc_ext{d_size}_{units_size}units_delay_tau{tau_unit}')
    errs = np.array(errs).T
    res = np.hstack((unit_idxs, errs))
    FileController.export_data(res, coords_file, title=f'err_ext{d_size}_{units_size}units_delay_tau{tau_unit}')

    idx_over10p = []
    for step in range(time_step):
        if np.max(accs[step, 1:]) - accs[step, 0] >= 0.1:
            idx_over10p.append(step)

    # export data
    accs_over10p = np.vstack([accs[i] for i in idx_over10p])
    units_over10p = np.vstack([unit_idxs[i] for i in idx_over10p])
    res = np.hstack((units_over10p, accs_over10p))
    FileController.export_data(res, coords_file, title=f'acc_ext{d_size}_{units_size}units_delay_tau{tau_unit}_over10p')
    # export graph
    graph = GraphViewer(fig_size=[12, 8])
    graph.boxplot(accs_over10p, ((np.arange(d_size + 1) + 1) * units_size).tolist(),
                  f'{coords_file.split("/")[-1][:-4]}_acc_ext{d_size}_delay_tau{tau_unit}_over10p',
                  r'$\tau=5$', 'Number of units', 'Accuracy rate')

    # export data
    errs_over10p = np.vstack([errs[i] for i in idx_over10p])
    res = np.hstack((units_over10p, errs_over10p))
    FileController.export_data(res, coords_file, title=f'err_ext{d_size}_{units_size}units_delay_tau{tau_unit}_over10p')
    # export graph
    graph = GraphViewer(fig_size=[12, 8])
    graph.boxplot(errs_over10p, ((np.arange(d_size + 1) + 1) * units_size).tolist(),
                  f'{coords_file.split("/")[-1][:-4]}_err_ext{d_size}_delay_tau{tau_unit}_over10p',
                  r'$\tau=5$', 'Number of units', 'cross entropy error')


def correlation_between_time_series():
    coords_file, (reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length) = import_data()

    np.random.seed(32)
    seed(32)

    steps = 1000
    units_size = 4
    corrs, accs = [], []

    csv_file = FileController.get_file(['csv'], './data_out')
    data = np.loadtxt(csv_file[0], delimiter=',')

    for step in trange(steps):
        units, accuracy = data[step, :units_size], data[step, units_size]
        tmp_corrs = np.array([pearsonr(X[:, int(s1)], X[:, int(s2)]) for s1, s2 in combinations(units, 2)])[:, 0]
        # min_corrs = np.min(np.abs(tmp_corrs))
        # corrs.append(min_corrs)
        mean_corrs = np.mean(np.abs(tmp_corrs))
        corrs.append(mean_corrs)
        accs.append(accuracy)
    corrs = np.array(corrs)
    accs = np.array(accs)

    slope, intercept, info = Calculate.linear_regression(accs.flatten(), corrs.flatten())

    graph = GraphViewer(fig_size=[12, 8])
    graph.scatter(accs, corrs, f'{coords_file.split("/")[-1][:-4]}_corr_mean_bw{units_size}', slope=slope,
                  intercept=intercept, title='Correlation between time series of 4 units', xlabel='error',
                  ylabel='correlation')


def show_attractor():
    """Image generation for the attractor

    """
    coords_file, (reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length) = import_data()
    # Accuracy rate is the highest with these 3 units
    units = [97, 156, 183]
    unit_idx = [i - 1 for i in units]
    graph = GraphViewer(fig_size=[12, 8], font_size=24, tick_size=16, is_constrained=False)
    graph.scatter3d_each_class(X, unit_idx, f'{coords_file.split("/")[-1][:-4]}',
                               xlabel=f'unit {units[0]}', ylabel=f'unit {units[1]}', zlabel=f'unit {units[2]}')


def untrained():
    """Response to unlearned data

    """
    coords_file, (reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length) = import_data()

    # Override 5-class classification settings.
    class_size = 5
    # Enter the index of the class to be unlearned（0-5）.
    drop_class = int(input('Input missed class id: '))
    label = label[:class_size * T, :class_size]
    target_data = np.zeros(reservoir_size)
    target_test_data = None
    for i in range(6):
        if i == drop_class:
            target_test_data = test_data[i * test_T:(i + 1) * test_T]
        else:
            target_data = np.vstack((target_data, X[i * T:(i + 1) * T]))
    X = target_data[1:]
    test_data = target_test_data

    # Ridge Regression (reg_param is 1e6 by default)
    Wout = Calculate.ridge_regression(X, label)
    # uncomment to save Wout as csv file
    # FileController.export_data(Wout, coords_file, title='_wout')

    stride, window = 1, 11
    averaged_test_data = Calculate.moving_average(test_data, window, stride)

    Y = np.dot(Wout, averaged_test_data.T).T

    figure_size = np.array([12, 8])
    font_size = 28
    legend_size = 22
    tick_size = 22
    anchor_options = [0, 1, 1, 0.15]
    plt.rcParams['font.size'] = font_size
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['font.sans-serif'] = ['Arial']

    custom_colors = [f'C{i}' for i in np.arange(6)]
    custom_colors.pop(drop_class)
    class_label = np.arange(6).tolist()
    class_label.pop(drop_class)

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot()

    for i in range(class_size):
        if class_label[i] % 2 == drop_class % 2:
            ax.plot(Y[:, i], linewidth=3, color=custom_colors[i])
        else:
            ax.plot(Y[:, i], linewidth=1, color=custom_colors[i])
    ax.set_xlabel('Step', fontsize=font_size)
    ax.set_ylabel('Value', fontsize=font_size)
    ax.legend([f'class {l}' for l in class_label], ncol=3, fontsize=legend_size,
              bbox_to_anchor=anchor_options, borderaxespad=0, frameon=False)
    ax.tick_params(labelsize=tick_size)

    dt = datetime.today().isoformat().replace(':', '-').split('.')[0]
    prev_image_title = f'{dt}_{coords_file.split("/")[-1][:-4]}'
    fig.savefig(f'./image_out/{prev_image_title}_reg_dropcls{drop_class}.eps', format='eps')


def robustness():
    """Verification of Robustness when Gradually Cutting Wout

    """
    coords_file, (reservoir_size, T, test_T, class_size, X, test_data, label, test_label, test_length) = import_data()

    # Ridge Regression (reg_param is 1e6 by default)
    Wout = Calculate.ridge_regression(X, label)
    accs = []
    res = []
    # Index of the class to be used as a reference
    for class_id in range(class_size):
        Wout_dropped = np.copy(Wout)
        for j in range(166 // 2 - 1):
            # Select units to drop.
            bool = sorted(np.abs(Wout_dropped[class_id, :]))[-1] <= np.abs(Wout_dropped[class_id, :])
            fp = np.where(bool == True)[0][0] % (reservoir_size // 2)
            # drop a unit.
            Wout_dropped[:, fp] = 0
            Wout_dropped[:, fp + reservoir_size // 2] = 0

            stride, window = 1, 11
            averaged_test_data = Calculate.moving_average(test_data, window, stride)

            Y_dropped = np.dot(Wout_dropped, averaged_test_data.T).T
            classified_dropped = np.argmax(Y_dropped, axis=1)
            wrong_count_dropped = np.count_nonzero(
                classified_dropped - np.hstack([np.ones(test_T) * i for i in range(class_size)])[5:-5])
            accuracy_rate_dropped = 1 - wrong_count_dropped / (test_T * class_size)
            accs.append(accuracy_rate_dropped)
            res.append([accuracy_rate_dropped, fp, fp + reservoir_size // 2])
    accs = np.array(accs)
    res = np.array(res)
    FileController.export_data(res, coords_file, title='robust')

    figure_size = np.array([12, 8])
    font_size = 28
    legend_size = 22
    tick_size = 22
    anchor_options = [0, 1, 1, 0.15]
    plt.rcParams['font.size'] = font_size
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot()
    for i in range(6):
        ax.plot((np.arange(166 // 2 - 1) + 1) * 2, accs[i * (166 // 2 - 1):(i + 1) * (166 // 2 - 1)], color=f'C{i}')
    ax.set_xlabel('Number of removed reservoir units', fontsize=font_size)
    ax.set_ylabel('Accuracy', fontsize=font_size)
    ax.tick_params(labelsize=tick_size)
    ax.legend([f'class {i}' for i in range(6)], ncol=3, fontsize=legend_size,
              bbox_to_anchor=anchor_options, borderaxespad=0, frameon=False)
    dt = datetime.today().isoformat().replace(':', '-').split('.')[0]
    prev_image_title = f'{dt}_{coords_file.split("/")[-1][:-4]}'
    fig.savefig(f'./image_out/{prev_image_title}_cut.eps', format='eps')


def main():
    commands = [f'{i}: {f}' for i, f in enumerate([
        'Classification with 6 class',
        'Classification with 24 class',
        'Relation between Number of units and accuracy',
        'Accuracy of classify',
        'Relation between fp amplitude variance and error',
        'Relation between Number of units and accuracy by logistic regression',
        'delay',
        'delay expansion',
        'correlation between time series',
        'show attractor',
        'untrained',
        'robustness'
    ])]
    features = [
        regression_6class,
        regression_24class,
        relation_between_unit_size_and_accuracy,
        accuracy_of_classify,
        relation_between_fp_amplitude_variance_and_error,
        relation_between_unit_size_and_accuracy_by_logistic,
        delay,
        delay_expansion,
        correlation_between_time_series,
        show_attractor,
        untrained,
        robustness
    ]
    text = '\n'.join(commands)
    feature = int(input(f'{text}\n\nSelect the function to use: '))

    return features[feature]()


if __name__ == '__main__':
    main()
