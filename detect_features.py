import cv2
from utils import FileController


def main():
    has_capture = input('Capture from camera? [y/n]: ') == 'y'

    device_id = 0
    selected_video = None if has_capture else FileController.get_file(['mp4'], './video_in')[0]
    capture_from = device_id if has_capture else selected_video
    video_capture = cv2.VideoCapture(capture_from)

    has_save = input('Save a video? [y/n]: ') == 'y'
    video_name = input('Input new video\'s name: ') if has_save else selected_video
    has_save_first_frame = input('Save a first frame? [y/n]: ') == 'y'

    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # export .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(f'./video_out/{video_name}_fp.mp4', fourcc, fps,
                                   (width, height)) if has_save else None

    detectors_name = [
        'AgastFeatureDetector',
        'FAST',
        'MSER',
        'AKAZE',
        'BRISK',
        'KAZE',
        'Oriented FAST and Rotated BRIEF',
        'SimpleBlobDetector'
    ]
    detectors = [
        cv2.AgastFeatureDetector_create,
        cv2.FastFeatureDetector_create,
        cv2.MSER_create,
        cv2.AKAZE_create,
        cv2.BRISK_create,
        cv2.KAZE_create,
        cv2.ORB_create,
        cv2.SimpleBlobDetector_create
    ]
    text = '\n'.join([f'{i}: {name}' for i, name in enumerate(detectors_name)])
    feature = int(input(f'{text}\n\nSelect the method of feature detection: '))
    detector = detectors[feature]()
    cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)

    print('*Note: press ESC key if you want to stop the video')
    while 1:
        retval, frame = video_capture.read()
        if retval is False:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect features
        keypoints = detector.detect(frame_gray)
        # draw keypoints on the frame
        new_frame = cv2.drawKeypoints(frame_gray, keypoints, None)

        if has_save_first_frame:
            cv2.imwrite(f'./image_out/{detectors_name[feature]}_frame1.png', new_frame)
            has_save_first_frame = False

        cv2.imshow('frame', new_frame)
        print('Start the video')

        if video_writer is not None:
            video_writer.write(new_frame)

        k = cv2.waitKey(30) & 0xff
        # ESC キーで終了
        if k == 27:
            print('Pressed ESC key and stopped the video')
            break

    cv2.destroyAllWindows()
    video_capture.release()
    if video_writer is not None:
        video_writer.release()


if __name__ == '__main__':
    main()
