import cv2
import numpy as np
from utils import FileController


def main():
    has_capture = input('Capture from camera? [y/n]: ') == 'y'

    device_id = 0
    selected_video = None if has_capture else FileController.get_file(['mp4'], './video_in')[0]
    capture_from = device_id if has_capture else selected_video
    video_capture = cv2.VideoCapture(capture_from)

    has_save_first_frame = input('Save a first frame? [y/n]: ') == 'y'

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=10, blockSize=7)

    color = np.random.randint(0, 255, (100, 3))

    # take first frame and find corners in it
    retval, frame = video_capture.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
    corners = np.int0(corners)

    retval, frame = video_capture.read()

    cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)

    for corner in corners:
        i, j = corner.ravel()
        cv2.circle(frame, (i, j), 5, (200, 0, 0), -1)
    cv2.imshow('frame', frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if has_save_first_frame:
        cv2.imwrite(f'./image_out/detect_gftt.png', frame_gray)

    video_capture.release()


if __name__ == '__main__':
    main()
