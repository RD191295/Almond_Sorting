from __future__ import print_function
import cv2
import numpy as np
import imutils

flag_detected = 0
Red_Counters = 0
"""
def rescale_frame(frame, percent=80):  # make the video windows a bit smaller
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)"""

img = cv2.imread("5.jpg")
print("for image 5")
# ANIMESH EDIT
down_height = 750
down_width = 1100
down_points = (down_width, down_height)
frame = cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)
#
# frame = rescale_frame(img)
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

count=[]
soakedcount = []
drycount = []
badcount = []
ldry=1
lsoaked=1
lbad=1

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
#cv2.imshow("Dry", red)
contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# dilation of the image ( to remove noise) create mask for dry almond
dry = cv2.dilate(dry, kernal, iterations=1)
dry1 = cv2.bitwise_and(frame, frame, mask=dry)
#cv2.imshow("Soaked", dry)
contours1, hierarchy = cv2.findContours(dry, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
soakedlist = []
brokenlist = []

for contour in contours1:
    x, y, w, h = cv2.boundingRect(contour)
    c = cv2.contourArea(contour)
    if (c >= 500 and c <= 6500 ):
        #frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        brokenlist.append(contour)
        #soaked
        #soakedcount.append(frame)
        #lsoaked=len(soakedcount)
        #print("soaked",lsoaked)

        #cv2.putText(frame, "Broken", (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    elif (c>=3500):
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        soakedlist.append(contour)
        cv2.putText(frame, "Soaked", (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    c = cv2.contourArea(contour)
    if (c >= 6500):
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        #for dry
        drycount.append(frame)
        ldry=len(drycount)
        #print("dry", ldry)

        cv2.putText(frame, "dry", (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif (c >= 500 and c <= 6500):
        brokenlist.append(contour)
        #frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        #for bad
        #badcount.append(frame)
        #lbad=len(badcount)
        #print("bad",lbad)
        

        #cv2.putText(frame, "Broken11", (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
for broken in brokenlist:
    x, y, w, h = cv2.boundingRect(broken)
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    cv2.putText(frame, "Broken", (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

total_count= ldry+lbad+lsoaked
print( "Total Almond={} , Soaked Almond={}, Dry Almond={},Broken Almond={}".format(total_count,lsoaked,ldry,lbad))


Dsoaked= lsoaked/total_count
persoaked=Dsoaked *100
format_persoaked = '{:.2f}'.format(persoaked)
print(format_persoaked)
cv2.putText(frame,  "Soaked Almond ={} %".format(format_persoaked), (20,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


Ddry= ldry/total_count
perdry=Ddry *100
format_perdry = '{:.2f}'.format(perdry)
print(format_perdry)
cv2.putText(frame,  "Dry Almond ={} %".format(format_perdry), (20,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

Dbad= lbad/total_count
perbad=Dbad *100
format_perbad = '{:.2f}'.format(perbad)
print(format_perbad)
cv2.putText(frame,  "Broken Almond ={} %".format(format_perbad), (20,85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


cv2.imshow("output", frame)
cv2.waitKey(0)



