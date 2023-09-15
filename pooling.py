from tkinter import filedialog
from tkinter import *
import cv2
from numba import jit
import numpy as np
from tqdm import tqdm


def average_filter(block):
    return


def maximum_filter(block: np.ndarray) -> np.ndarray:
    if block.ndim == 2:
        return np.max(block)
    elif block.ndim == 3:
        return np.max(np.max(block, axis=0), axis=0)


@jit(nopython=True)
def pooling(image: np.ndarray, in_h: int, in_w: int, out_h: int, out_w: int, in_d: int, wd: int = 2,
            st: int = 2) -> np.ndarray:
    new_image = np.zeros((out_h, out_w, in_d), dtype=np.uint8)
    for idx_h, h in enumerate(range(0, in_h, st)):
        for idx_w, w in enumerate(range(0, in_w, st)):
            for d in range(in_d):
                new_image[idx_h, idx_w, d] = np.max(image[h:h + wd, w:w + wd, d])

    return new_image


if __name__ == '__main__':
    file_type = (('mp4 file', '*.mp4'), ('mov file', '*.mov'))
    init_dir = './video_in'
    root = Tk()
    root.filename = filedialog.askopenfilename(filetypes=file_type, initialdir=init_dir)
    selected = root.filename
    root.destroy()

    window = 3
    stride = window

    video_capture = cv2.VideoCapture(selected)
    in_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = (in_width - window) // stride + 1
    out_height = (in_height - window) // stride + 1
    depth = 3
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(f'{selected[:-4]}_avg_pl.mp4', fourcc, fps, (out_width, out_height))
    print(f'input: {selected}, width: {out_width}px, height: {out_height}px, FPS: {fps}fps')

    min = float(input('How long? (minutes): '))
    # fps * second * minute
    time = np.ceil(60 * 60 * min).astype(np.int)

    for i in tqdm(range(time)):
        retval, frame = video_capture.read()
        if not retval:
            break

        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_frame = pooling(frame, in_height, in_width, out_height, out_width, depth, window, stride)
        video_writer.write(new_frame)

    cv2.destroyAllWindows()
    video_capture.release()
    video_writer.release()
