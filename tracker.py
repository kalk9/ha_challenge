"""
import necessary libraries
"""
from collections import OrderedDict
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment


class Tracker():
    """
    Tracker class that tracks objects based on the euclidean distance between the centers of the bounding boxes in a new frame and the reference frame
    """

    def __init__(self):
        """
            Func: initialize variables
        """
        self.nextID = 0                 # ID of object in the new frame
        self.objects = OrderedDict()    # ordered dictionary keeping track of detected objects
        self.lost = OrderedDict()       # ordered dictionary keeping track of lost objects that have missing detections
        self.maxLost = 10               # set maximum allowed missing detections to 10 frames
    
    def addObject(self, bbox_center ):
        """
            Func: Add newly found object to our list of tracked objects
            Args:
                bbox_center (tuple):  (x1, y1)   center of the object's bounding box
        """
        self.objects[self.nextID] = bbox_center     # assing for each object a new ID. The objects are defined by the center of their bbox
        self.lost[self.nextID] = 0                  # set the number of lost frames for this object to be 0
        self.nextID += 1                            

    def removeObject(self, id):
        """
            Func: Remove object from our tracked list when not found after frames lost > self.maxLost
            Args: 
                id (int): lost object's id
        """
        del self.objects[id]
        del self.lost[id]
    
    def euclideanDistance(self, ref_pts, new_pts, ids):
        """
            Function: Compute the euclidean distance between each bounding box center point in the current frame and each one in the reference or previous frame
            Args:
                ref_pts (list): list of bounding box center points in the reference frame
                new_pts (list): list of bounding box center points in the current frame
                ids (list): list of ids for all the tracked objects
            Returns:
                distance (2D array): matrix containing the euclidean distances computed
                id_pairs (3D array): matrix that associated the tracked ID with the detection index of the object found in current frame
        """
        distance = np.zeros((ref_pts.shape[0], new_pts.shape[0]))
        id_pairs = np.zeros((ref_pts.shape[0], new_pts.shape[0], 2))
        for i in range(0, distance.shape[0]):
            for j in range(0, distance.shape[1]):
                distance[i][j] = np.sqrt((new_pts[j][0]-ref_pts[i][0])**2 + (new_pts[j][1]-ref_pts[i][1])**2)
                id_pairs [i][j][0] = ids[i]
                id_pairs [i][j][1] = j
        return distance, id_pairs

    def hungarianAlgorithm (self, detections, distances, id_pairs):
        """
            Func: applies the Hungarian Algorithm to match new detections to tracked objects
            Args: 
                detections (list): List of bounding boxes detected in current frame 
                distances (2D array): matrix containing the euclidean distances between each bounding box center in the current frame and each bounding box center in the previous frame 
                id_pairs (2d array): matrix that associated the correct tracked ID with the detection index of the object
            Returns: 
                real_matches (list): list of matched new detections and tracked objects 
                unmatched_decetions (list): list of unmatched newly detected objects
                unmatched_objects (list): list of unmatched previously tracked objects
        """

        matched_idx = linear_assignment(distances)		# apply the hungarian algorithm to the distance matrix to find optimal assignment

        # initialize lists to keep track of unmatched previously tracked objects and unmatched new detections 
        unmatched_objects, unmatched_detections = [], []    
        # initalize lists to keep tracked of matche dobjects
        matches, real_matches = [], []

        for m in matched_idx:
            matches.append(m.reshape(1,2))

        if (len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        # real matches are found in the id_pairs matrix which contains the correct ID numbers of detections and tracked objects
        for match in matches:
            real_matches.append(np.array([int(id_pairs[match[0]][match[1]][0]), int(id_pairs[match[0]][match[1]][1])])) 

        # find unmatched detections and objects
        for (id,obj) in (self.objects.items()):
            if real_matches:
                if (id not in np.array(real_matches)[:,0]):
                    unmatched_objects.append(id)
            else:
                    unmatched_objects.append(id)

        for (id, det) in enumerate(detections):
            if real_matches:
                if id not in np.array(real_matches)[:,1]:
                    unmatched_detections.append(id)
            else:
                    unmatched_detections.append(id)
        
        return real_matches, np.array(unmatched_detections), np.array(unmatched_objects)


    def track(self, detections):
        """
            Func: Track objects
            Args:
                detections (list): List of bounding boxes detected in current frame 
            Returns:
                objects (OrderedDict): ordered dictionary containing tracked objects and assigned ids
        """
        # if no detections to match tracked objects with, mark existing tracked objects as lost and return objects
        if len(detections) == 0:
            for id in list(self.lost.keys()):
                self.lost[id] += 1
                # if object has been lost for over maxLost frames, stop tracking the object
                if self.lost[id] > self.maxLost:
                    self.removeObject(id)
            return self.objects

        # initialize and populate array containing bounding box centers of new detection bounding boxes
        centers = np.zeros((len(detections), 2), dtype="int")
        for (i, (x1, y1, w, h)) in enumerate(detections):
            center_x = int((x1 + (x1+w)) / 2)
            center_y = int((y1 + (y1+h)) / 2)
            centers[i] = (center_x, center_y)

        # if we have no objects that are tracked from previous frames, add new detections to the list of objects we are tracking
        if len(self.objects) == 0:
            for i in range(0, len(centers)):
                self.addObject(centers[i])

        # otherwise, match the new detected objects with previosuly tracked objects using the Hungarian Algorithm
        else:
            ids = list(self.objects.keys())                             # list of tracked objects IDs
            Object_centers = np.array(list(self.objects.values()))      # list of tracked objects bbox centers

            # compute the euclidean distances between bbox centers of tracked objects and new detections
            distance, id_pairs = self.euclideanDistance(Object_centers, centers, ids)   

            # call the Hungarian Algorithm to match detections with tracked objects
            real_matches, unmatched_detections, unmatched_objects = self.hungarianAlgorithm(detections, distance, id_pairs)
        
            # for all matched detections update value of tracked object and set lost frames to 0
            for match in (np.array(real_matches)):
                self.objects[match[0]] = centers[match[1]]
                self.lost[match[0]] = 0

            # for all unmatched previously tracked objects update value of lost frames
            for unmatched_obj in (unmatched_objects):
                self.lost[unmatched_obj] +=1
                # if object has been lost for over maxLost frames, stop tracking object by removing it from our dictionary of objects
                if self.lost[unmatched_obj]>self.maxLost:
                    self.removeObject(unmatched_obj)
                    np.delete(unmatched_objects, np.where(unmatched_objects == unmatched_obj))

            #for all unmatched new detections, start tracking them by add them to our dictionary of tracked objects
            for unmatched_det in (unmatched_detections):	
                self.addObject(centers[int(unmatched_det)])
        
        return self.objects