import cv2
import numpy as np
from utils import FileController

# .mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


def concatenate_video() -> None:
    video_count = int(input('How many videos do you input?: '))
    file_types = ['mp4' for i in range(video_count)]
    init_dir = './video_in'
    video_files = FileController.get_file(file_types, init_dir)

    cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)

    video_writer = None
    title = input('video name to save: ')
    for video in video_files:
        print(video)
        video_capture = cv2.VideoCapture(video)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        if video_writer is None:
            video_writer = cv2.VideoWriter(f'./video_in/{title}.mp4', fourcc, fps, (width, height))

        for i in range(np.ceil(fps).astype(np.int) * 60 * 22):
            retval, frame = video_capture.read()
            if not retval:
                break

            video_writer.write(frame)


def get_video_length():
    file_type = ['mp4']
    init_dir = './video_in'
    video_file = FileController.get_file(file_type, init_dir)
    video_capture = cv2.VideoCapture(video_file.pop(0))

    about_length = int(input('about length: '))
    for i in range(about_length):
        retval, frame = video_capture.read()
        if not retval:
            print(f'video length: {i}')
            break


def separate_video():
    file_type = ['mp4']
    init_dir = './video_in'
    video_file = FileController.get_file(file_type, init_dir)

    video_capture = cv2.VideoCapture(video_file.pop(0))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    # 5 min
    video_length = np.ceil(fps) * 60 * 5

    start_frame = [599, 19462, 38325, 57604, 76587, 95330, 114913, 133712, 152636, 172398, 191197, 210060, 229403,
                   248266, 267309, 286768, 305691, 324674, 345092, 363895, 382698, 402217, 421080, 439884]
    end_frames = (np.array(start_frame) + video_length).astype(np.int).tolist()
    titles = [f'0524_a{(i * 45):03}_w{j * 3 + 1}' for i in range(8) for j in range(3)]

    video_writer = None
    count = 0
    for i in range(60 * 60 * 128):
        retval, frame = video_capture.read()
        if not retval:
            break

        if i < start_frame[count]:
            continue
        elif i == start_frame[count]:
            video_writer = cv2.VideoWriter(f'./video_in/{titles[count]}.mp4', fourcc, fps, (width, height))
            print(f'title: {titles[count]}, step: {i}')
        elif start_frame[count] < i < end_frames[count]:
            video_writer.write(frame)
        elif i == end_frames[count]:
            video_writer.write(frame)
            video_writer.release()
            count += 1
            if count == len(start_frame):
                break
        elif end_frames[count] < i:
            continue

    video_capture.release()


if __name__ == '__main__':
    commands = [f'{i + 1}: {f}' for i, f in
                enumerate(['concatenate_video', 'get_video_length', 'separate_video'])]
    text = '\n'.join(commands)
    num = input(f'{text}\n\nSelect the function to use: ')

    if num == '1':
        concatenate_video()
    elif num == '2':
        get_video_length()
    elif num == '3':
        separate_video()
