import numpy as np
import cv2
import time

path_handmodel = "./FandRec/deephand/miohands.caffemodel"
path_handproto = "./FandRec/deephand/submit-net.prototxt"
net = cv2.dnn.readNetFromCaffe(path_handproto, path_handmodel)

r_img = cv2.imread("thumb.jpg")
img = cv2.resize(r_img, (224, 224))
h,w = r_img.shape[:2]
blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224),
                             (104.0, 177.0, 123.0), False)

net.setInput(blob)
detected_hands = net.forward()
print(detected_hands)
print(detected_hands.shape)
for i in range(0, detected_hands.shape[1]):
    confidence = detected_hands[0, 0, i, 2]
    if confidence > .5:
        box = detected_hands[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        conf_text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(img, conf_text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
cv2.imshow("Output", img)
cv2.waitKey(1)

