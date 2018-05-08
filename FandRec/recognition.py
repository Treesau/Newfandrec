import sys, cv2, dlib, numpy as np, os, time
from pathlib import Path
from gesture import *
from database import DBHelper
DBHelper = DBHelper.DBHelper


class Recognition:
    """
    
    """
    # global class variables
    hand_cascade = cv2.CascadeClassifier('./haarcascades/aGest.xml')
    gesture_recognizer = HandGestureRecognition()
    font = cv2.FONT_HERSHEY_SIMPLEX
    sample_size = 100
    gesture_timeout = 3
    rec_trained = False

    
    def __init__(self):
        """Class constructor
        """
        # initialize OpenCV's local binary pattern histogram recognizer
        # and load saved training data
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if Path('./trainingData/recognizer.yml').is_file():
            self.recognizer.read('./trainingData/recognizer.yml')
            self.rec_trained = True
            

        # instance variables
        self.frame_info = None
        self.samples = 0
        self.sample_images = []
        self.gesture_start = 0
        self.roi = None
        self.last_gest = ""
        self.is_tracking = True
        
        # bools for altering webpage control flow
        self.is_registering = False
        self.reg_complete = False
        

    def processFrame(self, frame, username):
        """
        
        """
        
        # split raw frame into color, grayscale
        bgr = cv2.UMat(frame)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        gesture = '0'  # default value will not send tag
        
        # reg_complete bool indicates whether registration has been completed
        # during this method call
        self.reg_complete = False
        
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
            faces = self._combineFaces(gray)
            if len(faces) > 1:
                pass
            else:
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x, y),
                                  (x + w , y + h),
                                  (255,0,0), 2)
                    self.samples += 1
                    gray = cv2.UMat(gray, [y,y+h], [x,x+w])
                    gray = cv2.resize(gray, (100, 100))
                    self.sample_images.append(gray)
                    
        else:   # finished collecting face data
            db = DBHelper()
            user_id = db.getIDByUsername(username)
            id_array = [user_id] * self.sample_size
            self.sample_images = self.sample_images[0:self.sample_size]
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
        out_frame = frame
        username = ""
        gesture = "0"
        num_fingers = 0

        if self.is_tracking:
            
            faces = self._combineFaces(gray)
            for (x,y,w,h) in faces:

                user_id, confidence = self.recognizer.predict(cv2.UMat(gray,[y,y+h],[x,x+w]))
                    
                if confidence <= 80:
                    db = DBHelper()
                    username = db.getUsernameById(user_id)
                    cv2.rectangle(out_frame, (x, y),
                                  (x + w, y + h),
                                  (225,105,65), 2)
                    cv2.putText(out_frame, username,
                                (int(x + 5), int(y - 10)),
                                self.font, 1.5,
                                (225,105,65), 2)

                    hands = self.hand_cascade.detectMultiScale(gray, 1.1, 7)
                    for (x,y,w,h) in hands:
                        
                        # draw rectangle around detected hand
                        cv2.rectangle(out_frame,(x,y),(x+w,y+h),(0,255,0),2)
                        
                        # calculate center point of detected hand
                        c_x = w//2 + x
                        c_y = h//2 + y
                        
                        # calculate region of interest by building a rectangle around detected hand
                        s = 2
                        f_x = int(c_x - w*s)
                        f_y = int(c_y - h*s)
                        f_w = int(w*s)
                        f_h = int(h*s)
                        self.roi = (f_x,f_y,f_w,f_h)
                        self.is_tracking = False

                        # begin timing gesture
                        self.gesture_start = time.time()
                else:
                    cv2.rectangle(out_frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(out_frame, "unknown",
                                (int(x + 5), int(y - 10)),
                                self.font, 1.5,
                                (0, 0, 255), 2)
                    
                
        else:
            x,y,w,h = self.roi
            depth = cv2.UMat.get(depth)
            try:
                num_fingers, out_hand = self.gesture_recognizer.recognize(depth[y:y+h,x:x+w])
                out_frame = cv2.UMat.get(out_frame)
                out_frame[y:y+h,x+x_w] = out_hand
                cv2.putText(out_frame, num_fingers,
                            (int(x + 5), int(y - 10)),
                            self.font, 1.5,
                            (0, 0, 255), 2)
            except:
                pass
            if (time.time() - self.gesture_start) > self.gesture_timeout:
                gesture = str(num_fingers)
                self.last_gest = gesture
                self.is_tracking = True

        return (out_frame, username, gesture)
    

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
                    (0,255,0), 2)
        return frame
    

    def _reset(self):
        """
        reinitializes variables used in gesture detection state
        """
        self.is_registering = False
        self.samples = 0
        self.sample_images = []
        self.roi = None
        self.is_tracking = True

    
        
        
                
    
