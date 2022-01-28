# IMPORT LIBRARY
from __future__ import print_function
import cv2
import numpy as np
import imutils

img = cv2.imread("1.jpg")

down_height = 750
down_width = 1100
down_points = (down_width, down_height)
frame = cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)

out_new = np.uint8(frame)
out_Gray = cv2.cvtColor(out_new, cv2.COLOR_BGR2GRAY)
ret, thresh_out = cv2.threshold(out_Gray, 127, 255, cv2.THRESH_BINARY_INV)
kernel_ip = np.ones((2, 2), np.uint8)
eroded_ip = cv2.erode(thresh_out, kernel_ip, iterations=1)
dilated_ip = cv2.dilate(eroded_ip, kernel_ip, iterations=1)
cnts = cv2.findContours(dilated_ip.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

if len(cnts) == 0:
    flag_empty = 1
    flag_detected = 0
    cv2.imshow("output", frame)
    cv2.waitKey(30)

# converting  BGR to HSV Frame
Big_faulty = max(cnts, key=cv2.contourArea)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# lIST TO STORE SIZE OF BOUNDING BOX (X,Y,W,H)
soked_bounding = []
broken_bounding = []
dry_bounding = []

# the range of dry almond
blu_lower = np.array([0, 120, 190], np.uint8)
blu_upper = np.array([255, 255, 255], np.uint8)

# the range of soaked almond
dry_lower = np.array([16, 0, 213], np.uint8)
dry_upper = np.array([255, 255, 255], np.uint8)

# finding the range of dry and soaked in the image
red = cv2.inRange(hsv, blu_lower, blu_upper)
dry = cv2.inRange(hsv, dry_lower, dry_upper)

kernal = np.ones((3, 3), "uint8")

# dilation of the image ( to remove noise) create mask for red color
red = cv2.dilate(red, kernal, iterations=1)
res = cv2.bitwise_and(frame, frame, mask=red)

contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# dilation of the image ( to remove noise) create mask for dry almond
dry = cv2.dilate(dry, kernal, iterations=1)
dry1 = cv2.bitwise_and(frame, frame, mask=dry)

# noinspection PyRedeclaration
contours1, hierarchy = cv2.findContours(dry, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours1:
    c = cv2.contourArea(contour)
    if c >= 3500:
        x, y, w, h = cv2.boundingRect(contour)
        soked_bounding.append([x, y, w, h])

for contour in contours:
    c = cv2.contourArea(contour)
    if c >= 6500:
        x, y, w, h = cv2.boundingRect(contour)
        dry_bounding.append([x, y, w, h])

    elif 500 <= c <= 6500:
        x, y, w, h = cv2.boundingRect(contour)
        broken_bounding.append([x, y, w, h])

"""
IMAGE 1 : put difference of soked- broken as like this : 30,50,50,50
          put difference of dry -broken as like this : 80,100,100,100

IMAGE 5 : put difference of soked- broken as like this : 70,110,70,70
          put difference of dry -broken as like this : 50,30,110,30
"""

# REMOVE DUPLICATE ( HERE WE MAKE BROKEN AS SUPER CLASS AND REMOVE INSIDE BOX OR NEAR ONE )
for soaked in soked_bounding:
    for broken in broken_bounding:
        if abs(soaked[0] - broken[0]) <= 30:
            if abs(soaked[1] - broken[1]) <= 50:
                if abs(soaked[2] - broken[2]) <= 50:
                    if abs(soaked[3] - broken[3]) <= 50:
                        if soaked in soked_bounding:
                            soked_bounding.remove(soaked)
        else:
            pass

for dry in dry_bounding:
    for broken in broken_bounding:
        if abs(dry[0] - broken[0]) <= 80:
            if abs(dry[1] - broken[1]) <= 100:
                if abs(dry[2] - broken[2]) <= 100:
                    if abs(dry[3] - broken[3]) <= 100:
                        print(dry)
                        dry_bounding.remove(dry)
        else:
            pass

# CREATE BOUNDING BOX
for soked in soked_bounding:
    cv2.rectangle(frame, (soked[0], soked[1]), (soked[0] + soked[2], soked[1] + soked[3]), (255, 0, 0), 2)
    cv2.putText(frame, "soked", (soked[0] - 5, soked[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

for broken in broken_bounding:
    cv2.rectangle(frame, (broken[0], broken[1]), (broken[0] + broken[2], broken[1] + broken[3]), (0, 0, 255), 2)
    cv2.putText(frame, "broken", (broken[0] - 5, broken[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

for dry in dry_bounding:
    cv2.rectangle(frame, (dry[0], dry[1]), (dry[0] + dry[2], dry[1] + dry[3]), (0, 255, 0), 2)
    cv2.putText(frame, "dry", (dry[0] - 5, dry[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("output", frame)
cv2.waitKey(0)