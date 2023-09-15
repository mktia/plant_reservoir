import copy
import csv
import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import FileController


def main():
    """Feature point tracking using Shi-Tomasi Corner Detector ＆ Optical Flow.
    Concatenate and track multiple videos in input order.
    
    """
    # Real-time processing of video being captured is possible if specified by device_id.
    # USB cameras are also supported, so be careful to select device_id when running on a PC equipped with a webcam.
    # device_id = 0
    # has_capture = input('Capture from camera? [y/n]: ') == 'y'
    has_save = input('Save a video? [y/n]: ') == 'y'
    title = input('title: ') if has_save else ''

    # Video Selection Using Dialog
    # num_of_video = int(input('How many videos?: ')) if is_capture != 'y' else 1
    # file_types = ['mp4' for i in range(num_of_video)]
    # init_dir = './video_in'
    # selected_video = FileController.get_file(file_types, init_dir) if is_capture != 'y' else [device_id]

    # 24 class
    selected_video = ['./video_in/0524_a000_w1.mp4',
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

    # Video Output Codec
    # .mov
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None

    # params for ShiTomasi corner detection
    # maxCorners:Maximum number of detections，qualityLevel:Threshold to be considered a feature point，minDistance:Minimum distance between feature points，blockSize:Search Window Size
    feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=20, blockSize=7)
    # params for lucas kanade optical flow
    # winSize:Search Window Size，maxLevel:Pyramids Hierarchy
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0, 255, (100, 3))

    cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)

    xs = []
    ys = []
    st = []
    num_of_features = 0
    num_of_frame = 0
    end_frames = []
    is_initialized = False

    prev_frame_gray, corners, mask = None, None, None
    for capture_from in selected_video:
        video_capture = cv2.VideoCapture(capture_from)

        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print(f'input: {capture_from}, width: {width}px, height: {height}px, FPS: {fps}fps, frame: {num_of_frame}')
        if video_writer is None and has_save:
            video_writer = cv2.VideoWriter(f'video_out/{title}.mp4', fourcc, fps, (width, height))

        # take first frame and find corners in it
        if num_of_frame == 0:
            retval, previous_frame = video_capture.read()
            prev_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(prev_frame_gray, mask=None, **feature_params)
            # create a mask image for drawing purposes
            mask = np.zeros_like(previous_frame)

        while 1:
            retval, frame = video_capture.read()
            if retval is False:
                end_frames.append(num_of_frame)
                break

            # convert frame from RGB into gray scale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            feature, status, track_error = cv2.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, corners, None,
                                                                    **lk_params)

            # Ignore quality of feature points
            good_new = feature
            good_old = corners

            # Consider quality of feature points (feature points disappear in the process)
            # good_new = feature[status == 1]
            # good_old = corners[status == 1]

            st.append([i for lst in status for i in lst])

            if not is_initialized:
                num_of_features = len(good_new)
                print(f'Get {num_of_features} points')
                xs = [[] for i in range(num_of_features)]
                ys = [[] for i in range(num_of_features)]
                is_initialized = True

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                # to flat ndarray
                a, b = new.ravel()
                c, d = old.ravel()

                xs[i].append(c)
                ys[i].append(d)

                # mask = cv2.line(mask, (a, b), (c, d), color[i % 100].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 3, color[i % 100].tolist(), -1)
                frame = cv2.putText(frame, f'{i + 1}', (a, b), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
            image = cv2.add(frame, mask)

            # Real-time display if video is not saved
            if video_writer is not None:
                video_writer.write(image)
            else:
                cv2.imshow('frame', image)

                k = cv2.waitKey(30) & 0xff
                # ESC key to exit
                if k == 27:
                    print('press ESC key')
                    break

            # new update teh previous frame and previous point
            prev_frame_gray = frame_gray.copy()
            corners = good_new.reshape(-1, 1, 2)
            num_of_frame += 1
    print(end_frames)

    cv2.destroyAllWindows()
    video_capture.release()
    if video_writer is not None:
        video_writer.release()

    xs = np.array(xs)
    ys = np.array(ys)
    xs_T = xs.T
    ys_T = ys.T

    # Status information for each feature point 0/1 (1 if the quality of the feature point is above the threshold, 0 if it is below)
    with open(f'./data_in/{title}_st.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(st)

    # Output x,y coordinates of feature points
    with open(f'./data_in/{title}.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        data = np.hstack((xs_T, ys_T))
        writer.writerows(data)


if __name__ == '__main__':
    main()
