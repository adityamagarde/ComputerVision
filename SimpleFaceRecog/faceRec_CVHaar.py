#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 18:52:13 2018

@author: adityamagarde
"""

#Importing Libraries
import cv2

#Loading cascades
faceCascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(r'haarcascade_eye.xml')

#Doing the detection
def detectFace(grayFrame, frame):
    faces = faceCascade.detectMultiScale(grayFrame, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 3)
        regionOfInterestGray = grayFrame[y:y+h, x:x+w]
        regionOfInterestColor = frame[y:y+h, x:x+w]
        
        eyes = eyeCascade.detectMultiScale(regionOfInterestGray, 1.1, 15)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(regionOfInterestColor, (ex, ey), (ex+ew, ey+eh), (0,255,255), 2)
    
    return frame

#Using the webcam and doing the detection
videoCapture = cv2.VideoCapture(0)
while True:
    _, frame = videoCapture.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detectFace(grayFrame, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()