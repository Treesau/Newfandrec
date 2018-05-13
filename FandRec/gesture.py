import cv2
import numpy as np
import math

class HandGestureRecognition:
    """
    """
    
    def __init__(self):
        """
        """
        self.kernel = kernel = np.ones((3,3),np.uint8)
        self.angle_cuttoff = 80.0

    def recognize(self, img):
        """
        """
        segment = self._segmentHand(img)
        try:
            contours, defects = self._findHullDefects(segment)
            return self._detectGesture(contours, defects, img)
        except:
            segment = cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR)
            return (0,segment)

    def _segmentHand(self, img):
        """
        """
        mask = cv2.erode(img, self.kernel, iterations = 2)
        mask = cv2.dilate(mask, self.kernel, iterations = 2)
        mask = cv2.GaussianBlur(mask,(5,5),0)
        return mask
        
    def _findHullDefects(self, segment):
        """
        """
        _,contours,hierarchy = cv2.findContours(segment,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key = lambda x: cv2.contourArea(x))
        epsilon = 0.01*cv2.arcLength(max_contour, True)
        max_contour = cv2.approxPolyDP(max_contour, epsilon, True)

        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)

        return (max_contour, defects)

    def _detectGesture(self, contours, defects, img):
        """
        """
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if defects is None:
            return ['0', img]

        if len(defects) <= 2:
            return ['0', img]

        num_fingers = 1

        for i in range(defects.shape[0]):
            start_idx, end_idx, farthest_idx, _ = defects[i,0]
            start = tuple(contours[start_idx][0])
            end = tuple(contours[end_idx][0])
            far = tuple(contours[farthest_idx][0])

            cv2.line(img, start, end, [0, 255, 0], 2)

            if angleRad(np.subtract(start, far),
                        np.subtract(end, far)) < deg2Rad(self.angle_cuttoff):
                num_fingers += 1

                # draw point as green
                cv2.circle(img, far, 5, [0, 255, 0], -1)
            else:
                # draw point as red
                cv2.circle(img, far, 5, [0, 0, 255], -1)

        return (min(5, num_fingers), img)

def angleRad(v1, v2):
    """Convert degrees to radians
    This method converts an angle in radians e[0,2*np.pi) into degrees
    e[0,360)
    """
    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
    
def deg2Rad(angle_deg):
    """Angle in radians between two vectors
    returns the angle (in radians) between two array-like vectors
    """
    return angle_deg/180.0*np.pi
    
