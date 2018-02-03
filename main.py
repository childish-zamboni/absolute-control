import numpy as np
import cv2
import argparse
from collections import deque

cap = cv2.VideoCapture(0)

pts = deque(maxlen = 64)

lower = (110,50,50)
upper = (130,255,255)

while True:
    ret, img=cap.read()
    img = cv2.flip(img, 1)

    writing = np.zeros((300, 300, 3), np.uint8);

    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Convert to HSV
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask= cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel) #Remove noise
    mask = cv2.dilate(mask, kernel, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 5:
            cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 0), 2)
            cv2.circle(img, center, 5, (0, 255, 0), 1)

    pts.appendleft(center)

    #Displaying
    for i in range (1, len(pts)):
        if pts[i-1]is None or pts[i] is None:
            continue
        width = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)
        cv2.line(img, pts[i-1],pts[i],(0,0,225), width)
        cv2.line(writing, pts[i-1],pts[i],(0,0,225), 2)

    cv2.imshow("Frame", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Writing", writing)

    if cv2.waitKey(30) & 0xFF == 32:
        break

cap.release()
cv2.destroyAllWindows()
