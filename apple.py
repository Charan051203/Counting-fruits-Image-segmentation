import cv2
import numpy as np

lowerBound = np.array([170, 100, 80])
upperBound = np.array([180, 256, 256])

cam = cv2.VideoCapture(0)  # Fix: cam, not cap
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    img = cv2.imread('apple.jpg')

    # Convert BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV', imgHSV)

    # Create the Mask
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)

    # Morphology
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    maskFinal = maskClose

    # Fix: OpenCV 4.x returns only 2 values
    conts, _ = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, conts, -1, (255, 0, 0), 1)

    for i in range(len(conts)):
        x, y, w, h = cv2.boundingRect(conts[i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, str(i + 1), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

    cv2.imshow("maskClose", maskClose)
    cv2.imshow("maskOpen", maskOpen)
    cv2.imshow("mask", mask)
    cv2.imshow("cam", img)

    # Fix: Proper exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cam.release()  
        cv2.destroyAllWindows()
        break