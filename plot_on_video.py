"""
Woutを動画上に出力する
"""
from datetime import date
from tkinter import filedialog
from tkinter import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import FileController


def main():
    is_save = True if input('Is save a video? [y/n]: ') == 'y' else False
    file_types = ['mp4' for i in range(int(input('How many videos input?: ')))]
    init_dir = './video_in'
    # video_files = FileController.get_file(file_types, init_dir)
    # video_files = ['./video_in/P2200783_M_L_1_max_pl.mp4',
    #                './video_in/P2200786_M_R_1_max_pl.mp4',
    #                './video_in/P2200784_M_L_2_max_pl.mp4',
    #                './video_in/P2200787_M_R_2_max_pl.mp4',
    #                './video_in/P2200785_M_L_3_max_pl.mp4',
    #                './video_in/P2200788_M_R_3_max_pl.mp4']
    video_files = ['./video_in/0524_a000_w1.mp4',
                   './video_in/0524_a000_w4.mp4',
                   './video_in/0524_a000_w7.mp4',
                   './video_in/0524_a045_w1.mp4',
                   './video_in/0524_a045_w4.mp4',
                   './video_in/0524_a045_w7.mp4',
                   './video_in/0524_a090_w1.mp4',
                   './video_in/0524_a090_w4.mp4',
                   './video_in/0524_a090_w7.mp4',
                   './video_in/0524_a135_w1.mp4',
                   './video_in/0524_a135_w4.mp4',
                   './video_in/0524_a135_w7.mp4',
                   './video_in/0524_a180_w1.mp4',
                   './video_in/0524_a180_w4.mp4',
                   './video_in/0524_a180_w7.mp4',
                   './video_in/0524_a225_w1.mp4',
                   './video_in/0524_a225_w4.mp4',
                   './video_in/0524_a225_w7.mp4',
                   './video_in/0524_a270_w1.mp4',
                   './video_in/0524_a270_w4.mp4',
                   './video_in/0524_a270_w7.mp4',
                   './video_in/0524_a315_w1.mp4',
                   './video_in/0524_a315_w4.mp4',
                   './video_in/0524_a315_w7.mp4']
    file_types = ['csv' for i in range(2)]
    init_dir = './data_in'
    coords_file, wout_file = FileController.get_file(file_types, init_dir)
    # coords_file = './data_in/1111_l123r123_q1_md20_ws15_fixed.csv'
    # wout_file = './data_out/1111_l123r123_q1_md20_ws15_fixed_std_wout_nemoto.csv'

    coords_data = np.loadtxt(coords_file, delimiter=',').astype(np.float32)
    wout_data = np.loadtxt(wout_file, delimiter=',')
    _, reservoir_size = coords_data.shape
    class_size, _ = wout_data.shape
    # class_size = 6
    point_size = reservoir_size // 2

    video_capture = None
    video_writer = None
    color_map = plt.get_cmap('tab10')
    default_colors = [[int(e * 255) for e in color_map(i % 10)[:3]][::-1] for i in range(100)]
    white_color = (255, 255, 255)

    # l123r123
    coords_data = np.vstack((coords_data[:9315], coords_data[27585:36900], coords_data[9315:18360],
                             coords_data[36900:46245], coords_data[18360:27585], coords_data[46245:]))

    start = 0
    # lr1lr2lr3
    end_frames = [9315, 18630, 27675, 37020, 46245, 55830]
    # l123r123
    # end_frames = [9314, 18359, 27584, 36899, 46244, 55829]
    # end_frames = [60 * 60 * 5 * (i + 1) - 1 for i in range(24)]
    wout_xs = wout_data[:, :point_size]
    wout_ys = wout_data[:, point_size:]

    # .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)

    # num_of_fp = 92
    # units = [int(i) % num_of_fp for i in input('use reservoir units: ').split()]

    for cls, (video_file, end) in enumerate(zip(video_files, end_frames)):
        # for video_file, end in zip(video_files, end_frames):
        video_capture = cv2.VideoCapture(video_file)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        if video_writer is None and is_save:
            today = date.today()
            title = f'{"{:02d}{:02d}".format(today.month, today.day)}_{coords_file.split("/")[-1][:-4]}_wout'
            video_name = f'./video_out/{title}.mp4'
            video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        xs_data = coords_data[start:end, :point_size]
        ys_data = coords_data[start:end, point_size:]

        for xs, ys in tqdm(zip(xs_data, ys_data)):
            retval, frame = video_capture.read()

            # for i in range(class_size):
            i = cls
            for idx, (x, y, wout_x, wout_y) in enumerate(zip(xs, ys, wout_xs[i], wout_ys[i])):
                # if idx in units:
                # radius = ((np.abs(wout_x) + np.abs(wout_y)) * 1e3 * 4).astype(np.float32)
                # frame = cv2.circle(frame, (x, y), radius, default_colors[i])
                frame = cv2.circle(frame, (x, y), 3, default_colors[1], -1)
                frame = cv2.putText(frame, f'{idx + 1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, white_color)

            # todo delete
            cv2.imwrite('./image_out/video_fp_2.png', frame)
            exit()

            cv2.imshow('frame', frame)

            if video_writer is not None:
                video_writer.write(frame)
            else:
                cv2.imshow('frame', frame)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

        start = end

    video_capture.release()
    if video_writer is not None:
        video_writer.release()


if __name__ == '__main__':
    main()
