import numpy as np
import cv2
import time

path_handmodel = "./FandRec/caffe_models/hand_classifier.caffemodel"
path_handproto = "./FandRec/caffe_models/hand_classifier.prototxt"
net = cv2.dnn.readNetFromCaffe(path_handproto, path_handmodel)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    h,w = frame.shape[:2]
    blob = cv2.dnn.blobFromframe(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detected_hands = net.forward()

    for i in range(0, detected_hands.shape[2]):
        confidence = detected_hands[0, 0, i, 2]
        if confidence > .5:
            box = detected_hands[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            conf_text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, conf_text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
    cv2.imshow("Output", frame)
    cv2.waitKey(1)
