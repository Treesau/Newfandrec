import sys, cv2, numpy as np, os, time, dlib, imutils
from pathlib import Path
from gesture import *
from database import DBHelper
from imutils.video import WebcamVideoStream
DBHelper = DBHelper.DBHelper


class BackgroundModel:
    def __init__(self):
        self.calibrated = False
        self.background = None
        self.num_frames = 0

    def runAverage(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        if self.num_frames < 30:
            if self.background is None:
                self.background = gray.copy().astype("float")
            cv2.accumulateWeighted(gray, self.background, 0.5)
            self.num_frames += 1
        else:
            self.calibrated = True

class Recognition:
    """
    
    """
    # global class variables
    
    path_facemodel = "./models/face_classifier.caffemodel"
    path_faceproto = "./models/face_classifier.prototxt.txt"
    facenet = cv2.dnn.readNetFromCaffe(path_faceproto, path_facemodel)
    hand_classifier = cv2.CascadeClassifier("./models/aGest.xml")
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
        self.gesture_tracker = None
        self.last_gest = ""
        self.rec_trained = False
        self.black_mask = np.zeros((480,640),np.uint8)
        self.bg_model = BackgroundModel()
        
        
        # bools for altering webpage control flow
        self.is_registering = False
        self.reg_complete = False
        
        # initialize OpenCV's local binary pattern histogram recognizer
        # and load saved training data
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if Path('./training_data/recognizer.yml').is_file():
            self.recognizer.read('./training_data/recognizer.yml')
            self.rec_trained = True
        

    def processFrame(self, frame, username):
        """
        
        """
        # get frame height and width
        if self.frame_dimensions is None:
            self.frame_dimensions = frame.shape[:2]
        
        # split raw frame into color, grayscale
        frame = cv2.UMat(frame)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        frame = cv2.UMat.get(frame)
        
        # reg_complete bool indicates whether registration has been completed
        # during this method call
        self.reg_complete = False

        gesture = '0'  # default value will not send tag
        
        if self.is_registering:
            frame = self._register(frame, gray, username)
            self._displayProgress(frame)
        elif self.rec_trained:
            frame, username, gesture = self._detect(frame, gray)
            self._displayGesture(frame)
        
        return (frame, username, gesture)
        
    
    def _register(self, frame, gray, username):
        """
        """ 
        if self.samples < self.sample_size:

            faces = self._findFaces(frame)
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
                    w = c_w
                    h = c_h
                    maxArea = w*h

            if faces:
                self.samples += 1
                gray = cv2.UMat(gray,[y,y+h],[x,x+w])
                gray = cv2.resize(gray, (100, 100))
                self.sample_images.append(gray)
                    
        else:   # finished collecting face data
            db = DBHelper()
            user_id = db.getIDByUsername(username)
            id_array = [user_id] * self.sample_size
            
            for i in range(self.sample_size):
                self.sample_images[i] = cv2.UMat.get(self.sample_images[i])
	    
            if Path('./training_data/recognizer.yml').is_file():
                self.recognizer.update(self.sample_images, np.array(id_array))
            else:
                self.recognizer.train(self.sample_images, np.array(id_array))
            self.recognizer.write('./training_data/recognizer.yml')
            
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
        
        if self.gesture_tracker is None:
            faces = self._findFaces(frame)
            for (startX,startY,endX,endY) in faces:
                
                y = startY - 10 if startY - 10 > 10 else startY + 10
                try:
                    gray_face = cv2.UMat(gray,[startY,endY],[startX,endX])
                except:
                    gray_face = gray[startY:endY,startX:endX]
                gray_face = cv2.resize(gray_face, (100, 100))
                user_id, confidence = self.recognizer.predict(gray_face)
                gray = cv2.UMat.get(gray)
                gray[startY:endY,startX:endX] = self.black_mask[startY:endY,startX:endX]
                if confidence <= 80:
                    db = DBHelper()
                    username = db.getUsernameById(user_id)
                    cv2.putText(frame, username,
                                (startX, y),
                                self.font, .6,
                                (225,105,65), 2)
                else:
                    cv2.putText(frame, "unknown",
                                (startX, y),
                                self.font, .6,
                                (0, 0, 255), 2)
                    
            if username is not "" and faces:
                hands = self.hand_classifier.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in hands:
                    x_mid = (w//2)
                    y = int(y-h*1.3)
                    x = int(x-x_mid*1.5)
                    w = int(w+3*x_mid)
                    h = int(h*2+h*0.7)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255), 2)
                    if self.bg_model.calibrated:
                        self.gesture_tracker = GestureTracker(frame,(x,y,w,h))
                    
            if not self.bg_model.calibrated and not faces:
                self.bg_model.runAverage(frame)

        else:
            timed_out, (x,y,w,h) = self.gesture_tracker.update(frame)
            if timed_out:
                self.gesture_tracker = None
            try:
                gray = cv2.UMat.get(gray)
                difference = cv2.absdiff(self.bg_model.background.astype("uint8")[y:y+h,x:x+w],
                                         gray[y:y+h,x:x+w])
                foreground = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
                gest, frame[y:y+h,x:x+w] = self.gesture_recognizer.recognize(foreground)
                self.last_gest = str(gest)
            except:
                pass
            

        return (frame, username, gesture)
    

    def _findFaces(self, frame):
        """
        """
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (150, 150)), 1.0,
                                     (150, 150), (104.0, 177.0, 123.0))
        self.facenet.setInput(blob)
        detected_faces = self.facenet.forward()
        h,w = self.frame_dimensions
        face_regions = []

        for i in range(0, detected_faces.shape[2]):
            confidence = detected_faces[0,0,i,2]
            
            if confidence > .4:
                box = detected_faces[0,0,i,3:7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")
                if (startX >= 0 and startY >= 0 and
                    endX <= w and endY <= h):

                    face_regions.append((startX, startY, endX, endY))
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (225,105,65), 2)

        return face_regions
    

    def _displayProgress(self, frame):
        """
        """
        x, y = self.frame_dimensions
        percent_complete = str(int(self.samples/self.sample_size*100)) + "%"
        cv2.putText(frame, percent_complete,
                    (25, 30), self.font, 1.2,
                    (0,255,0), 2)
        return frame


    def _displayGesture(self, frame):
        """
        """
        x, y = self.frame_dimensions
        message = "Gesture:  " + self.last_gest
        cv2.putText(frame, message,(25, 30),
                    self.font, 1.2,(225,105,65), 2)


    def _reset(self):
        """
        reinitializes variables used in gesture detection state
        """
        self.is_registering = False
        self.samples = 0
        self.sample_images = []

class GestureTracker:

    gesture_timeout = 4
    
    def __init__(self, frame, rect):
        self.timed_out = False
        self.start_time = time.time()
        self.corr_tracker = dlib.correlation_tracker()
        x,y,w,h = rect
        self.corr_tracker.start_track(frame, dlib.rectangle(x,y,
                                                            x+w,
                                                            y+h))

    def update(self, frame):
        
        tracking_quality = self.corr_tracker.update(frame)
        
        if (time.time() - self.start_time) >= self.gesture_timeout:
            self.timed_out = True
        
        tracked_position = self.corr_tracker.get_position()
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
    cam = WebcamVideoStream(src=0).start()
    user = ""
    
    response = input("Register new user? y or n \n")
    if response == 'y':
        rec.is_registering = True
        user = input("Enter a username: ")
        db.createUser([user, "", "", "", "", ""])
    else:
        rec.is_registering = False
    while (True):
        frame = cam.read()
        frame = cv2.resize(frame,(640,480))
        out, user, gest = rec.processFrame(frame, user)
        cv2.imshow("out", out)
        cv2.waitKey(1)
    cam.release()
    cv2.destroyAllWindows()
