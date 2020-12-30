# -*- coding: utf-8 -*-

import cv2
import numpy as np

#start video frame
frame = cv2.imread("IMG-20200819-WA0005.jpg")

# converting  BGR to HSV Frame
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#pilled almond range
white_lower = np.array([7, 100, 100], np.uint8)
white_upper = np.array([27, 255, 255], np.uint8)

    
# finding the range of white color in the image
white = cv2.inRange(hsv, white_lower, white_upper)
    

kernal = np.ones((5, 5), "uint8")
            
# dilation of the image ( to remove noise) create mask for white color
white = cv2.dilate(white, kernal, iterations=1)
res = cv2.bitwise_and(frame, frame, mask=white)
    
        
contours, hierarchy = cv2.findContours(white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 4000:  # if white color object size is grater than 1000 it will create reactangle area
        x, y, w, h = cv2.boundingRect(contour)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, "Pilled Almond", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

         
cv2.imwrite("test_IMG-20200819-WA0005.jpg",frame)