import numpy as np
import cv2
import utils as ut

cap = cv2.VideoCapture('video.avi')
nFrame = 0
T = 20
med = 100

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # this part is background initialisation according to ChangeDetection.pdf
    if nFrame < med:
        if nFrame == 0:
            prev = gray.copy()
        s = cv2.add(gray.astype(np.int32), prev.astype(np.int32))
        prev = s.copy()

    # this part is background subst and updating according to ChangeDetection.pdf
    else:
        if nFrame == med:
            bckI = (s/med).astype(np.uint8)

        substraction = cv2.absdiff(gray, bckI)
        #c_mask = cv2.threshold(substraction, T, 255, cv2.THRESH_BINARY)[1]

        c_mask = ut.subst(substraction, T)

        #Morphology
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 1]])
        morf = cv2.dilate(c_mask, kernel, iterations=0)
        kernel = np.array([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]])
        morf = cv2.morphologyEx(morf, cv2.MORPH_CLOSE, cv2.getStructuringElement(kernel))
        morf = cv2.morphologyEx(morf, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))


        # Blob Analysis

        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 0
        params.maxThreshold = 256

        params.filterByArea = True
        params.minArea = 500

        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        rv = 255 - morf
        keypoints = detector.detect(rv)

        # keypoints is a list of keypoint, which include the coordinates (of the centres) of the blobs, and their size

        im_with_keypoints = cv2.drawKeypoints(morf, keypoints, np.array([]), (0, 0, 155), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # End Blob Analysis
        ut.show(Bacground=bckI, Mask=c_mask, Morf=morf, Blob= im_with_keypoints)
        key = cv2.waitKey(0)
        bckU = ut.updating_background(morf, gray, bckI, 0.2)
        bckI = bckU.copy()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
