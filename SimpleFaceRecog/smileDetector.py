#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 20:07:18 2018

@author: adityamagarde
"""

#Importing Libraries
import cv2

#Loading cascades
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(r'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(r'haarcascade_smile.xml')

#Doing the detection
def detectSmile(frame, grayFrame):
    faces = faceCascade.detectMultiScale(grayFrame, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        regionOfInterestGray =  grayFrame[y:y+h, x:x+w]
        regionOfInterestColor = frame[y:y+h, x:x+w]
        
        smile = smileCascade.detectMultiScale(regionOfInterestGray, 1.1, 15)
        for(sx, sy, sw, sh) in smile:
            cv2.rectangle(regionOfInterestColor, (sx, sy), (sx+sw, sy+sh), (0,255,0), 2)
            
        return frame

#Using the webcam
videoCapture = cv2.VideoCapture(0)
while True:
    _, frame = videoCapture.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detectSmile(frame, grayFrame)
    cv2.imshow('Smile', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()