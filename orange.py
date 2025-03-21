import cv2
import numpy as np

lowerBound = np.array([1, 100, 80])
upperBound = np.array([22, 256, 256])

kernelOpen = np.ones((5,5))
kernelClose = np.ones((20,20))

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    img = cv2.imread('orange.png')
    
    # Convert BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create the Mask
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    
    # Morphology
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    maskFinal = maskClose
    
    # Fix: cv2.findContours now returns only 2 values in OpenCV 4+
    conts, _ = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    cv2.drawContours(img, conts, -1, (255, 0, 0), 3)
    
    for i in range(len(conts)):
        x, y, w, h = cv2.boundingRect(conts[i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, str(i + 1), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255))

    cv2.imshow("maskClose", maskClose)
    cv2.imshow("maskOpen", maskOpen)
    cv2.imshow("mask", mask)
    cv2.imshow("cam", img)
    
    # Fix: Use cv2.waitKey(1) for smoother response
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Breaks the loop properly

# Ensures all windows close properly
cv2.destroyAllWindows()
