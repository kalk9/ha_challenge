# ha_challenge
Humanising Autonomy Python Challenge

## Requirements to run the program
- Python 3.7.4
- numpy==1.21.6
- opencv_python==4.1.2.30
- scikit_learn==0.22.2

## Scripts
- draw_detections.py: draws and prints bounding boxes, detection labels, scores and tracking IDs on video frames
- tracker.py: multi-object tracker based on Euclidean Distance
- display_video.py: displays a video, it detections and tracked IDs

## To run the program
- python3 display_video.py --v_path <PATH_TO_VIDEO> --l_path <PATH_TO_LABELING>
- example: python3 display_video.py -v_path "resources/video_3.mp4" -l_path "resources/video_3_detections.json"

