import sys, cv2, numpy as np, os, time, dlib
from pathlib import Path
from gesture import *
from database import DBHelper
DBHelper = DBHelper.DBHelper


class Recognition:
    """
    
    """
    # global class variables
    
    path_facemodel = "./caffe_models/face_classifier.caffemodel"
    path_faceproto = "./caffe_models/face_classifier.prototxt.txt"
    facenet = cv2.dnn.readNetFromCaffe(path_faceproto, path_facemodel)
    
    hand_detector = dlib.simple_object_detector("detector.svm")
    gesture_recognizer = HandGestureRecognition()
    font = cv2.FONT_HERSHEY_SIMPLEX
    sample_size = 100

    
    def __init__(self):
        """Class constructor
        """
        # instance variables
        self.frame_dimensions = None
        self.samples = 0
        self.sample_images = []
        self.gest_trackers = []
        self.last_gest = ""
        rec_trained = False
        # bools for altering webpage control flow
        self.is_registering = False
        self.reg_complete = False
        
        # initialize OpenCV's local binary pattern histogram recognizer
        # and load saved training data
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if Path('./trainingData/recognizer.yml').is_file():
            self.recognizer.read('./trainingData/recognizer.yml')
            self.rec_trained = True
        

    def processFrame(self, frame, username):
        """
        
        """
        # get frame height and width
        if self.frame_dimensions is None:
            self.frame_dimensions = frame.shape[:2]
        
        # split raw frame into color, grayscale
        bgr = cv2.UMat(frame)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        
        # reg_complete bool indicates whether registration has been completed
        # during this method call
        self.reg_complete = False

        gesture = '0'  # default value will not send tag
        
        if self.is_registering:
            display = self._register(display, gray, username)
            display = self._displayProgress(display)
        elif self.rec_trained:
            display, username, gesture = self._detect(display, gray, depth)
            display = self._displayGesture(display)
        
        return (display, username, gesture)
        
    
    def _register(self, frame, gray, username):
        """
        """ 
        if self.samples < self.sample_size:

            frame, faces = self._findFaces(frame)
            
            max_area = 0
            x = 0
            y = 0
            w = 0
            h = 0
            
            for (startX,startY,endX,endY) in faces:
                c_w = endX - startX
                c_h = endY - startY
                if c_w * c_h > max_area:
                    x = startX
                    y = startY
                    w = _w
                    h = _h
                    maxArea = w*h
                    
            self.samples += 1
            gray = cv2.UMat(gray, [startY,endY], [startX,endX])
            gray = cv2.resize(gray, (100, 100))
            self.sample_images.append(gray)
                    
        else:   # finished collecting face data
            db = DBHelper()
            user_id = db.getIDByUsername(username)
            id_array = [user_id] * self.sample_size
            #self.sample_images = self.sample_images[0:self.sample_size]
            
            for i in range(self.sample_size):
                self.sample_images[i] = cv2.UMat.get(self.sample_images[i])
	    
            if Path('./trainingData/recognizer.yml').is_file():
                self.recognizer.update(self.sample_images, np.array(id_array))
            else:
                self.recognizer.train(self.sample_images, np.array(id_array))
            self.recognizer.write('./trainingData/recognizer.yml')
            
            # registration complete
            self.reg_complete = True
            self.rec_trained = True
            
            # reset variables before detection begins
            self._reset()
            
        return frame


    def _detect(self, frame, gray):
        """
            :param frame: a BGR color image for display
            :param gray: a grayscale copy of the passed BGR frame
            :returns: (out_frame, username, gesture) the processed frame
            for display on webpage, the detected user, the detected gesture
        """
        username = ""
        gesture = "0"
        num_fingers = 0

        if self.gesture_trackers:
            d_frame, faces = self._findFaces(frame)
            
            for (startX,startY,endX,endY) in faces:
                y = startY - 10 if startY - 10 > 10 else startY + 10
                gray = cv2.UMat(gray, [startY,endY], [startX,endX])
                gray = cv2.resize(gray, (100, 100))
                user_id, confidence = self.recognizer.predict(cv2.UMat(gray,
                                                                       [startY,endY],
                                                                       [startX,endX]))
                if confidence <= 80:
                    db = DBHelper()
                    username = db.getUsernameById(user_id)
                    cv2.putText(d_frame, username,
                                (startX, y),
                                self.font, .6,
                                (225,105,65), 2)

                    frame, hands = self._findHands(frame)
                    for (startX,startY,endX,endY) in hands:
                        tracked_hand = GestureTracker(frame,
                                                      (startX,startY,endX,endY))
                        self.gesture_trackers.append(tracked_hand)
                        
                else:
                    cv2.putText(d_frame, "unknown",
                                (startX, y),
                                self.font, .6,
                                (0, 0, 255), 2)   
                    
            return (d_frame, username, gesture)

        else:
            timed_out, roi = self.gesture_trackers[0].update(frame)
            if timed_out:
                self.gesture_trackers.clear()
            x,y,w,h = roi
            gest, processed_roi = self.gesture_recognizer.recognize(frame[y:y+h,x:x+w])
            #frame[startY-s:endY+s,startX-s:endX+s] = processed_roi
            self.last_gest = gest
    

    def _findFaces(self, frame):
        """
        """
        blob = cv2.dnn.blobFromframe(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
        self.facenet.setInput(blob)
        detected_faces = self.facenet.forward()
        h,w = self.frame_dimensions
        face_regions = []

        for i in range(0, detected_faces.shape[2]):
            confidence = detected_faces[0,0,i,2]
            
            if confidence > .5:
                box = detected_faces[0,0,i,3*7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")

                face_regions.append(startX, startY, endX, endY)
                
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (225,105,65), 2)

        return (frame, face_regions)
                

    def _findHands(self, frame):
        """
        """
        detections = self.hand_detector(frame)
        hand_rects = []
        
        for k, d in enumerate(hand_rects):
            hand_rects.append(d.left(), d.top(), d.right(), d.bottom())
            cv2.rectangle(frame, hand_rects(k),
                          (0, 0, 255), 2)

        return (frame, hand_rects)
    

    def _displayProgress(self, frame):
        """
        """
        x, y = self.frame_info.getColorRes()
        percent_complete = str(int(self.samples/self.sample_size*100)) + "%"
        cv2.putText(frame, percent_complete,
                    (x//2-5, y-10), self.font, 1.2,
                    (0,255,0), 2)
        return frame


    def _displayGesture(self, frame):
        """
        """
        x, y = self.frame_info.getColorRes()
        message = "Gesture:  " + self.last_gest
        cv2.putText(frame, message,
                    (x//4, y-10), self.font, 1.2,
                    (225,105,65), 2)
        return frame
    

    def _reset(self):
        """
        reinitializes variables used in gesture detection state
        """
        self.is_registering = False
        self.samples = 0
        self.sample_images = []

class GestureTracker:

    gesture_timeout = 3
    
    def __init__(self, frame, rect):
        self.timed_out = False
        self.start_time = time.time()
        self.corr_tracker = dlib.correlation_tracker()
        startX,startY,endX,endY = rect
        self.corr_tracker.start_track(frame, dlib.rectangle(startX, startY,
                                                            endX, endY))

    def update(self, frame):
        if (time.time() - self.start_time) >= self.gesture_timeout:
            self.timed_out = True
            
        tracking_quality = self.corr_tracker.update(frame)
        if tracking_quality >= 7:
            tracked_position = tracker.get_position()
            x = int(tracked_position.left())
            y = int(tracked_position.top())
            w = int(tracked_position.width())
            h = int(tracked_position.height())
            cv2.rectangle(frame, (x, y),
                          (x + w, y + h),
                          (0,255,0), 2)
            
            return (self.timed_out, (x,y,w,h))
        


if __name__ == "__main__":
    
    rec = Recognition()
    db = DBHelper()
    cam = cv2.VideoCapture(0)
    user = ""
    
    response = input("Register new user? y or n \n")
    if response == 'y':
        rec.is_registering = True
        user = input("Enter a username: ")
        db.createUser([user, "", "", ""])
    else:
        rec.is_registering = False
    while (True):
        ret, frame = cam.read()
        out, user, gest = rec.processFrame(frame, user)
        cv2.imshow("out", out)
        print(user + " " + gest)
        cv2.waitKey(1)
    cam.release()
    cv2.destroyAllWindows()

    
        
        
                
    
