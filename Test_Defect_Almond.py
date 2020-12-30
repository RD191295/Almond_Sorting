# -*- coding: utf-8 -*-
import cv2
import numpy as np

#start video frame
frame = cv2.imread("IMG-20200819-WA0012.jpg")

# converting  BGR to HSV Frame
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# the range of defect almond color
Defect_Lower = np.array([0, 0, 0], np.uint8)
Defect_Higher = np.array([50, 100, 100], np.uint8)

# finding the range of white color in the image
Defect = cv2.inRange(hsv, Defect_Lower, Defect_Higher)
    

kernal = np.ones((5, 5), "uint8")
            
# dilation of the image ( to remove noise) create mask for white color
Defect = cv2.dilate(Defect, kernal, iterations=1)
res = cv2.bitwise_and(frame, frame, mask=Defect)

        
contours, hierarchy = cv2.findContours(Defect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 4000:  # if white color object size is grater than 1000 it will create reactangle area
        x, y, w, h = cv2.boundingRect(contour)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(frame, "Defect Almond", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

         
cv2.imwrite("test_IMG-20200819-WA0012.jpg",frame)