import numpy as np
import cv2
import time

path_model = "face_pre.caffemodel"
path_caffe = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(path_caffe, path_model)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h,w = frame.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
faces = net.forward()
while True:
    ret, frame = cap.read()
    h,w = frame.shape[:2]
    frame = cv2.UMat(frame)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    for i in range(0, faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > .5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            conf_text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, conf_text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
    cv2.imshow("Output", frame)
    cv2.waitKey(1)
