"""
CSV編集
"""
from numba import jit
import numpy as np
from scipy.stats import zscore

from utils import Calculate, FileController


def calculate_moving_average() -> None:
    """移動平均データを作成する

    Returns:
        None
    """
    file_types = ['csv']
    init_dir = './data_in'
    coords_file = FileController.get_file(file_types, init_dir).pop(0)
    data = np.loadtxt(coords_file, delimiter=',')

    stride, window = 1, 11
    data_ma = Calculate.moving_average(data, window, stride)

    np.savetxt(f'{coords_file[:-4]}_ma.csv', data_ma, delimiter=',', fmt='%.5e')


def concatenate_csv() -> None:
    """CSVを連結する

    Returns:
        None
    """
    file_types = ['csv', 'csv']
    init_dir = './data_in'
    coords_files = FileController.get_file(file_types, init_dir)

    print('data: ' + ', '.join([i.split('/')[-1] for i in coords_files]))

    coords_data_1 = np.loadtxt(coords_files[0], delimiter=',')
    coords_data_2 = np.loadtxt(coords_files[1], delimiter=',')

    coords_data = np.vstack((coords_data_1[9314:], coords_data_2[9314:]))
    time_size, reservoir_size = coords_data.shape

    print(time_size)

    title = input('title: ')
    np.savetxt(f'./data_in/{title}.csv', coords_data, delimiter=',')


def create_label(class_size: int) -> None:
    """ラベル作成

    Returns:
        None
    """
    title = input('label name: ')
    video_length = 60 * 60 * 5
    zero = np.zeros((video_length, class_size))
    one = np.ones(video_length)
    matrix = np.ones(class_size)

    for i in range(class_size):
        tmp = np.copy(zero)
        tmp[:, i] = one
        matrix = np.vstack((matrix, tmp))
    matrix = matrix[1:]

    np.savetxt(f'./data_in/{title}_label.csv', matrix, delimiter=',', fmt='%d')


def drop_feature_points() -> None:
    """特徴点を消去する．ユニットの最初のインデックスは１（０スタートではない）

    Returns:
        None
    """
    file_types = ['csv']
    init_dir = './data_in'
    coords_file = FileController.get_file(file_types, init_dir).pop()

    coords_data = np.loadtxt(coords_file, delimiter=',')
    time_size, reservoir_size = coords_data.shape

    # 入力例（半角スペースで分割）61 68 80 75 9 37 60 84 22
    input_nums = np.array([int(i) for i in input('what numbers of feature points drop?: ').split()])
    input_nums = np.hstack((input_nums, input_nums + np.ones_like(input_nums) * reservoir_size // 2))
    input_nums -= np.ones_like(input_nums)

    zeros = np.zeros(time_size)

    for i in input_nums:
        coords_data[:, i] = zeros

    file_name = coords_file.split('/')[-1]
    np.savetxt(f'./data_in/{file_name[:-4]}_fixed.csv', coords_data, delimiter=',', fmt='%.5e')


def multiple_status() -> None:
    """追跡精度（ステータス）を反映させる

    Returns:
        None
    """
    file_types = ['csv', 'csv']
    init_dir = './data_in'
    coords_file, status_file = FileController.get_file(file_types, init_dir)

    data = np.loadtxt(coords_file, delimiter=',')
    status = np.loadtxt(status_file, delimiter=',')
    # for x+y
    data = data * np.hstack((status, status))

    np.savetxt(f'{coords_file[:-4]}_0.csv', data, delimiter=',')


def normalize() -> None:
    """[0, 1]区間で正規化する

    Returns:
        None
    """
    file_types = ['csv']
    init_dir = './data_in'
    coords_file = FileController.get_file(file_types, init_dir).pop(0)
    data = np.loadtxt(coords_file, delimiter=',')

    time_size, reservoir_size = data.shape
    for i in range(reservoir_size):
        if data[0, i] != 0:
            min = np.min(data[:, i])
            max = np.max(data[:, i])
            data[:, i] = (data[:, i] - min) / (max - min)

    np.savetxt(f'{coords_file[:-4]}_norm.csv', data, delimiter=',', fmt='%.5e')


def standardize() -> None:
    """平均を0にする

    Returns:
        None
    """
    file_types = ['csv']
    init_dir = './data_in'
    coords_file = FileController.get_file(file_types, init_dir).pop(0)
    data = np.loadtxt(coords_file, delimiter=',')

    time_size, reservoir_size = data.shape
    for i in range(reservoir_size):
        if data[0, i] != 0:
            data[:, i] = zscore(data[:, i])

    np.savetxt(f'{coords_file[:-4]}_std.csv', data, delimiter=',', fmt='%.5e')


def start_0() -> None:
    """時系列の開始を0に合わせる

    Returns:
        None
    """

    @jit(nopython=True)
    def minus(data) -> np.ndarray:
        data -= data[0] * np.ones_like(data)
        return data

    file_types = ['csv']
    init_dir = './data_in'
    coords_file = FileController.get_file(file_types, init_dir).pop()
    coords_data = np.loadtxt(coords_file, delimiter=',')

    # 初期座標を減算
    coords_data = minus(coords_data)
    np.savetxt(f'{coords_file[:-4]}_start0.csv', coords_data, delimiter=',', fmt='%.5e')


if __name__ == '__main__':
    # 使用する関数のコメントアウトを外す
    # calculate_moving_average()
    # concatenate_csv()
    # create_label(int(input('How many class?: ')))
    # drop_feature_points()
    # multiple_status()
    # normalize()
    # standardize()
    # start_0()
    pass
