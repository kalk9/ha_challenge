"""
import necessary libraries
"""
import cv2

class Draw():
    """"
    Class to draw and display bounding boxes, labels, scores and tracking IDs on video frames
    """

    def __init__(self, img: cv2.VideoCapture):
        """
        Func: Initialize variables
        Args: 
            img (cv2.VideoCapture): one video frame
        """

        self.img = img                                          # frame
        self.colors = {'bicycle': (241, 196, 15),               # for each label assign a different a color
        'bus': (231, 76, 60), 'car': (142, 68, 173), 
        'motorbike': (41, 128, 185), 
        'person': (39, 174, 96), 
        'truck': (44, 62, 80)}                   
    
    def draw_bbox(self, pt1: tuple, pt2: tuple, label: str, score: float):
        """
        Fuc: Function to draw bounding boxes, labels and scores
        Args: 
            pt1 (tuple): (x1, y1)       starting point of the bounding box (upper left corner)
            pt2 (tuple): (x2, y2)       ending point of the bounding box (lower right corner)
            label (str): detection label
            score (float): detection score
        Rteurns: frame with drawn detections (cv2.VideoCapture)
        """
        
        # draw bounding box around the detected object in the frame
        self.img = cv2.rectangle(self.img, pt1, pt2, color = self.colors[label], thickness = 2)
        # text background to display over top each bounding box, the class detected and the detection score
        text = "{}: {}".format(label, round(score,2))
        # find space required by the text
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # print the text on image on upper left corner of the bounding box    
        self.img = cv2.rectangle(self.img, (pt1[0], pt1[1] - 20), (pt1[0]+ w, pt1[1]), self.colors[label], -1)
        self.img = cv2.putText(self.img, text, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        return self.img

    def display_id(self, id: int, pos: tuple, label: str):
        """
        Func: Function to print tracking id for detected object
        Args:
            id (int): tracked object ID
            pos (tuple): (c_x1, c_x2)   center point of the bounding box
            label (str): detection label
        Returns: frame with IDs printed (cv2.VideoCapture)
        """
        
        # text displaying tracking ID
        text = "ID {}".format(id)
        # find space required by the text
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        #print the text in the center of the bounding box
        self.img = cv2.rectangle(self.img, (pos[0], pos[1] - 20), (pos[0]+ w, pos[1]), self.colors[label], -1)
        self.img = cv2.putText(self.img, text, (int(pos[0]), int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return self.img