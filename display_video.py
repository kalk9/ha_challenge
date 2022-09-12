"""
import necessary libraries
"""
from math import floor
# from typing import NoReturn, OrderedDict
from collections import OrderedDict
import json
import cv2
from tracker import Tracker
from draw_detections import Draw
import argparse

def open_video(path: str): # -> cv2.VideoCapture:
    """Opens a video file.

    Args:
        path: the location of the video file to be opened

    Returns:
        An opencv video capture file.
    """
    video_capture = cv2.VideoCapture(path)
    if not video_capture.isOpened():
        raise RuntimeError(f'Video at "{path}" cannot be opened.')
    return video_capture


def get_frame_dimensions(video_capture: cv2.VideoCapture):# -> tuple[int, int]:
    """Returns the frame dimension of the given video.

    Args:
        video_capture: an opencv video capture file.

    Returns:
        A tuple containing the height and width of the video frames.

    """
    return video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )


def get_frame_display_time(video_capture: cv2.VideoCapture): # -> int:
    """Returns the number of milliseconds each frame of a VideoCapture should be displayed.

    Args:
        video_capture: an opencv video capture file.

    Returns:
        The number of milliseconds each frame should be displayed for.
    """
    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    return floor(1000 / frames_per_second)


def is_window_open(title: str): # -> bool:
    """Checks to see if a window with the specified title is open."""

    # all attempts to get a window property return -1 if the window is closed
    return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1


def main(video_path: str, detections_dict: dict, title: str): # -> NoReturn:
    """Displays a video at half size until it is complete or the 'q' key is pressed.

    Args:
        video_path: the location of the video to be displayed
        title: the title to display in the video window
    """

    video_capture = open_video(video_path)
    width, height = get_frame_dimensions(video_capture)
    wait_time = get_frame_display_time(video_capture)
    

    try:
        # read the first frame
        frame_num = 1
        success, frame = video_capture.read()

        # initialize tracker
        tracker = Tracker()
        
        # create the window
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

        # run whilst there are frames and the window is still open
        while success and is_window_open(title):
            
            # get bounding boxes, scores and detections 
            frame_detections = detections_dict[str(frame_num)]
            bboxes = frame_detections["bounding boxes"]
            scores = frame_detections["detection scores"]
            classes = frame_detections["detected classes"]

            # initialize dictionary of tracked objects
            objects = OrderedDict()

            # only track pedestrians
            pedestrians_indices  = [index for (index, item) in enumerate(classes) if item == "person"]
            if pedestrians_indices:
                pedestrians_bboxes = [bboxes[i] for i in pedestrians_indices]
                objects = tracker.track(pedestrians_bboxes)


            detection_image = frame
            
            # initialize Draw object to draw and display bounding boxes, detections, scores and ids of detected objects
            draw = Draw(detection_image)
            # draw bounding boxes
            for (bbox, label, score) in zip(bboxes, classes, scores):
                detection_image = draw.draw_bbox( (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), label, score )
            
            # if we have tracked objects, display their ID
            if objects:
                for (id, center) in objects.items():
                    detection_image = draw.display_id(id, (center[0], center[1]), 'person')


            # shrink it
            smaller_image = cv2.resize(detection_image, (floor(width // 2), floor(height // 2)))
            # display it
            cv2.imshow(title, smaller_image)

            # test for quit key
            if cv2.waitKey(wait_time) == ord("q"):
                break

            # read the next frame
            success, frame = video_capture.read()
            frame_num = frame_num + 1
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='A program that detects and tracks objects')
    parser.add_argument("-v_path", "--video_path", help="Path of the video file", default="resources/video_3.mp4")
    parser.add_argument("-l_path", "--label_path", help="Path of the labeling file", default="resources/video_3_detections.json")
    args = parser.parse_args()

    # read detections from json file
    with open(args.label_path, 'r') as f:
        detections = json.load(f)
        
    main(args.video_path, detections, "My Video")
