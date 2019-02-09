import cv2
import numpy as np
import time

THRESHOLD = 600
FALSE_POSITIVE = 20000
PIXEL_INTENSITY_THRESHOLD = 20

def main():
    cap = cv2.VideoCapture(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.createBackgroundSubtractorMOG2()

    ret, frame1 = cap.read()

    while(cap.isOpened()):
        time.sleep(0.01)
        ret, frame2 = cap.read()

        # skin areas detection
        skinMask = HSVBin(frame1)
        contours = getContours(skinMask)
        ## Darw contours
        cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
        cv2.imshow('capture', frame1)

        # Create an image based on the differences between the two frames and then enhance the result
        diffImg = cv2.absdiff(frame1, frame2)
        threshImg = cv2.threshold(diffImg, PIXEL_INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

        # Our operations on the frame come here
        gray = cv2.cvtColor(threshImg, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # calculate white pixels amount
        whitePixels = cv2.findNonZero(fgmask)

        ## Draw the resulting frame for movement
        # cv2.imshow('frame', fgmask)

        if whitePixels is not None:
            amountOfMovement = len(whitePixels)
            if (amountOfMovement > THRESHOLD and amountOfMovement < FALSE_POSITIVE):
                movement = boundaries(amountOfMovement, 0, 100)
                print('movement: {}, countours: {}'.format(movement, len(contours)))

        frame1 = frame2
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def getContours(img):
    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    contours, h  = cv2.findContours(closed, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    validContours = [];
    for cont in contours:
        if cv2.contourArea(cont) > 9000:
            # x,y,w,h = cv2.boundingRect(cont)
            # if h/w > 0.75:
            validContours.append(cv2.convexHull(cont))
            # rect = cv2.minAreaRect(cont)
            # box = cv2.cv.BoxPoints(rect)
            # validContours.append(np.int0(box))
    return validContours

def HSVBin(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_skin = np.array([50, 50, 0])
    upper_skin = np.array([125, 125, 255])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # res = cv2.bitwise_and(img, img, mask=mask)
    return mask

def boundaries(number, mmin, mmax):
    scaled = int(number / mmax)
    if scaled < mmin:
        scaled = mmin
    if scaled > mmax:
        scaled = mmax
    return scaled

if __name__ == '__main__':
    main()