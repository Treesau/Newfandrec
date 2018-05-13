import numpy as np
import cv2
import time

path_facemodel = "./FandRec/caffe_models/face_classifier.caffemodel"
path_faceproto = "./FandRec/caffe_models/face_classifier.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(path_faceproto, path_facemodel)

full_img = cv2.imread("img2.jpg")
img = cv2.resize(full_img, (300, 300))

h,w = full_img.shape[:2]
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
faces = net.forward()
print(faces)
print(faces.shape)

for i in range(0, faces.shape[2]):
    confidence = faces[0, 0, i, 2]
    if confidence > .5:
        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        conf_text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(full_img, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(full_img, conf_text, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
cv2.imshow("Output", full_img)
cv2.waitKey(1)
