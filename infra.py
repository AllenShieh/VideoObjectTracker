# no bounding box + contour
# findContour()

# no bounding box + background subtraction + contour
# backgroundSubtraction()
# findContour()

# bounding box + contour
# subarea()
# findContour()

# bounding box + background subtraction + contour
# subarea()
# findContour()
import cv2
import numpy as np
from load_frameInfo import loadInfo
cap = cv2.VideoCapture("test.mp4")
fgbg = cv2.BackgroundSubtractorMOG()
lower_purple = np.array([165,50,90])
upper_purple = np.array([179,255,255])

ret, frame = cap.read()
Height, Width = frame.shape[:2]
frame_count = 0


allFrames = loadInfo("frameInfo.mat")
while True:
    if cap.grab():
        flag, frame = cap.retrieve()
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        boundingBoxes = allFrames[frame_count]
        mask = fgbg.apply(frame)

        # display bounding boxes
        frame1 = frame.copy()
        # background + contour
        frame2 = frame.copy()
        # subarea + background + contour
        frame3 = frame.copy()
        # frame4 background + subarea + contour
        frame4 = frame.copy()


        # frame2 background + contour
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cv2.drawContours(frame2, contours, -1, (0,255,0), 2)

        for i in range(len(boundingBoxes)):
            boundingBox = boundingBoxes[i]
            x = boundingBox[0] / 2
            y = boundingBox[1] / 2
            w = boundingBox[2] / 2
            h = boundingBox[3] / 2

            # frame1 display bounding boxes
            subArea = frame1[max(y, 0):min(Height, y+h), max(0, x): min(Width, x+w), 0:3]
            if (subArea.size > 0):
                subContours, _ = cv2.findContours((subArea[:,:,0]).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(subContours) > 0:
                    cv2.drawContours(subArea, subContours, -1, (0,255,0), 2)
                    frame1[max(y, 0):min(Height, y+h), max(0, x): min(Width, x+w), 0:3] = subArea

            # frame3 subarea + background + contour
            subArea = frame3[max(y, 0):min(Height, y+h), max(0, x): min(Width, x+w), 0:3]
            subMask = fgbg.apply(subArea)
            if (subArea.size > 0):
                subContours, _ = cv2.findContours(subMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(subContours) > 0:
                    cv2.drawContours(subArea, subContours, -1, (0,255,0), 2)
                    frame3[max(y, 0):min(Height, y+h), max(0, x): min(Width, x+w), 0:3] = subArea

            # frame4 background + subarea + contour
            subArea = frame4[max(y, 0):min(Height, y+h), max(0, x): min(Width, x+w), 0:3]
            subMask = mask[max(y, 0):min(Height, y+h), max(0, x): min(Width, x+w)]
            if (subMask.size > 0):
                subContours, _ = cv2.findContours(subMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(subContours) > 0:
                    cv2.drawContours(subArea, subContours, -1, (0,255,0), 2)
                    frame4[max(y, 0):min(Height, y+h), max(0, x): min(Width, x+w), 0:3] = subArea


        hmerge1 = np.hstack((frame1, frame2))
        hmerge2 = np.hstack((frame3, frame4))
        vmerge = np.vstack((hmerge1, hmerge2))
        cv2.imshow("result", vmerge)

        # cv2.imshow("no box + contour", frame1)
        # cv2.imshow("no box + background + contour", frame2)
        # cv2.imshow("box + contour", frame3)
        # cv2.imshow("box + background + contour", frame4)


    if cv2.waitKey(1) == 13: #13 is the Enter Key
            break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
