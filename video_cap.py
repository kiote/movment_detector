import numpy as np
import cv2
import time

THRESHOLD = 600

cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()

ret, frame1 = cap.read()

while(True):
    # Capture frame-by-frame
    ret, frame2 = cap.read()

    # Create an image based on the differences between the two frames and then enhance the result
    diffImg = cv2.absdiff(frame1, frame2)
    # threshImg = cv2.threshold(diffImg, PIXEL_INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # Our operations on the frame come here
    gray = cv2.cvtColor(diffImg, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

	# Assign frame 1 to frame 2 for the next iteration of comparison
    frame1 = frame2
 
    # Display the resulting frame
    cv2.imshow('frame', fgmask)

    whitePixels = cv2.findNonZero(fgmask)

    time.sleep(0.1)

    if whitePixels is not None:
        amountOfMovement = len(whitePixels)
        if (amountOfMovement > THRESHOLD):
            print(amountOfMovement)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()