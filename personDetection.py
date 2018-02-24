import cv2
import base64
import sys
from PIL import Image
import requests
import json
import time
import threading


videoCapture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
fullbodyCascade = cv2.CascadeClassifier('./haarcascade_fullbody.xml')
lowerbodyCascade = cv2.CascadeClassifier('././haarcascade_lowerbody.xml')
upperbodyCascade = cv2.CascadeClassifier('././haarcascade_upperbody.xml')
start = time.time()
latitude = 43.786766
longitude = -79.189723
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while(True):
    ret, frame = videoCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
    )
    fullBody = fullbodyCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    lowerBody = lowerbodyCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    upperBody = upperbodyCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    # (rects, weights) = hog.detectMultiScale(
    #     gray,
    #     winStride=(4,4),
    #     padding=(8,8),
    #     scale = 1.05,
    # )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in fullBody:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in lowerBody:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in upperBody:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # rects = np.array([[x,y,x+w,y+h] for (x, y, w, h) in rects])
    # pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # for (xA, yA, xB, yB) in pick:
    #     cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    currTime = time.time()
    elapsed = currTime - start
    if (elapsed >= 1):
        if (len(faces) != 0 or len(fullBody) != 0 or len(lowerBody) != 0 or len(upperBody) != 0 ):
            image = cv2.imencode('.png', frame)[1]
            print ("Detected Object")
            ########### Send Image ###############
            # cv2.imwrite("currImage.jpg", frame)
            # b64 = base64.b64encode(image)
            # data = {"image" : b64.decode("utf-8"),
            #         "latitude": latitude,
            #         "longitude": longitude}
            # headers = {'content-type': 'application/json'}
            # r = requests.post('https://sarhtv2.firebaseio.com/people.json', data=json.dumps(data), headers=headers)
        start = time.time()
        longitude = longitude + 0.0001
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()

