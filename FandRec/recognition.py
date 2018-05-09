import sys, cv2, numpy as np, os, time
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

    path_handmodel = "./caffe_models/hand_classifier.caffemodel"
    path_handproto = "./caffe_models/hand_classifier.prototxt"
    handnet = cv2.dnn.readNetFromCaffe(path_faceproto, path_facemodel)

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
            if len(faces) > 1:
                pass
            else:
                for (startX,startY,endX,endY) in faces:
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
                    
                    # calculate center point of detected hand
                    c_x = startX//2 + startX
                    c_y = startY//2 + startY

                    gest, d_hand = self.gesture_recognizer.recognize(depth[startY:endY,
                                                                           startX:endX])
                    d_frame[startY:endY,startX:endX] = d_hand
                    self.last_gest = gest

            else:
                cv2.putText(d_frame, "unknown",
                            (startX, y),
                            self.font, .6,
                            (0, 0, 255), 2)   
                
        return (d_frame, username, gesture)
    

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
        blob = cv2.dnn.blobFromframe(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
        self.handnet.setInput(blob)
        detected_hands = self.handnet.forward()
        h,w = self.frame_dimensions
        hand_regions = []

        for i in range(0, detected_hands.shape[2]):
            confidence = detected_hands[0,0,i,2]
            if confidence > .5:
                box = detected_hands[0,0,i,3*7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")

                hand_regions.append(startX, startY, endX, endY)
                
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)

        return (frame, hand_regions)
    

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

    
        
        
                
    
