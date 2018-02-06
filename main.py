#! python3
import pyautogui
import numpy as np
import cv2
import argparse
import time
import pytesseract
import threading
from collections import deque
from PIL import Image
import ctypes

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
tessdata_dir_config = '--tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata" --psm 10  --oem 2 '

kernel = np.ones((1, 1), np.uint8)
busy = False
rightMousePressed = False
leftMousePressed = False
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
handwriting = cv2.imread("media/temp.jpg")

fist_cascade = cv2.CascadeClassifier('classifiers/fist.xml')
palm_cascade = cv2.CascadeClassifier('classifiers/palm.xml')

user32 = ctypes.windll.user32
screenWidth = user32.GetSystemMetrics(0)
screenHeight = user32.GetSystemMetrics(1)

def read_handwriting():
    busy = False
    while True:
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
                    print(result[i])
                    pyautogui.press(result[i])
                    break;
        time.sleep(0.5)

t = threading.Thread(target=read_handwriting)
t.daemon = True
t.start()

cap = cv2.VideoCapture(0)

pts = deque(maxlen = 50)
lower = (110,50,50)
upper = (130,250,250)

while True:
    ret, img=cap.read()
    img = cv2.flip(img, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    writing = np.zeros(( int( cap.get(4) ), int (cap.get(3) ), 3), np.uint8);
    writing[:,:] = (255,255,255)

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
        pyautogui.moveTo(screenWidth*center[0]/cap.get(3), screenHeight*center[1]/cap.get(4))

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
        cv2.line(writing, pts[i-1],pts[i],(0, 0, 0), 5)

    if busy == False:
        busy = True
        cv2.imwrite('media/temp.jpg', writing)
        busy = False

    fists = fist_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in fists:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 255),2)

    if len(fists) != 0:
        if (fists[0][1] + fists[0][3]/2 >= cap.get(3)/2 and leftMousePressed == False):
            print('Left click!')
            if rightMousePressed:
                pyautogui.mouseUp(button='right')
            pyautogui.mouseDown(button='left')
            leftMousePressed = True;
            rightMousePressed = False;
        elif (fists[0][1] + fists[0][3]/2 < cap.get(3)/2 and rightMousePressed == False):
            print('Right click!')
            if leftMousePressed:
                pyautogui.mouseUp(button='left')
            pyautogui.mouseDown(button='right')
            rightMousePressed = True
            leftMousePressed = False
    else:
        if rightMousePressed:
            pyautogui.mouseUp(button='right')
            rightMousePressed = False
        if leftMousePressed:
            pyautogui.mouseUp(button='left')
            leftMousePressed = False

    cv2.imshow("Writing", writing)
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", img)

    if cv2.waitKey(30) & 0xFF == 32:
        break

cap.release()
cv2.destroyAllWindows()
