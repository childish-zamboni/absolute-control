import numpy as np
import cv2
import argparse
import time
import pytesseract
import threading
from collections import deque
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
tessdata_dir_config = '--tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata" --psm 10  --oem 2 '
kernel = np.ones((1, 1), np.uint8)
busy = False
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
handwriting = cv2.imread("media/temp.jpg")

def read_handwriting():
    busy = False
    while True:
        output = ''
        if busy == False:
            busy = True
            handwriting = cv2.imread("media/temp.jpg")
            busy = False

        handwriting = cv2.cvtColor(handwriting, cv2.COLOR_BGR2GRAY)
        handwriting = cv2.dilate(handwriting, kernel, iterations=1)
        handwriting = cv2.erode(handwriting, kernel, iterations=1)
        arr = Image.fromarray(handwriting)
        result = pytesseract.image_to_string(arr, config = tessdata_dir_config)
        for i in range(len(result) - 1):
            for c in range(len(characters)):
                if(result[i] == characters[c]):
                    output += result[i]
                    break;
        print(output)
        time.sleep(0.5)

t = threading.Thread(target=read_handwriting)
t.daemon = True
t.start()

cap = cv2.VideoCapture(0)

pts = deque(maxlen = 50)
lower = (110,50,50)
upper = (130,255,255)

while True:
    ret, img=cap.read()
    img = cv2.flip(img, 1)

    writing = np.zeros(( int( cap.get(4) ), int (cap.get(3) ), 3), np.uint8);
    writing[:,:] = (0,0,0)

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
        cv2.line(img, pts[i-1],pts[i],(0, 0, 225), width)
        cv2.line(writing, pts[i-1],pts[i],(255, 255, 225), 5)

    if busy == False:
        busy = True
        cv2.imwrite('media/temp.jpg', writing)
        busy = False

    cv2.imshow("Writing", writing)
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", img)

    if cv2.waitKey(30) & 0xFF == 32:
        break

cap.release()
cv2.destroyAllWindows()
