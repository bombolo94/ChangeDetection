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

        median = cv2.GaussianBlur(gray,(5,5),0)
        c_mask = cv2.threshold(cv2.absdiff(median,bckI), T, 255, cv2.THRESH_BINARY)[1]


        # Morphology
        morf = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12)))
        morf = cv2.morphologyEx(morf, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        # Blob Analysis

        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 0
        params.maxThreshold = 256

        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        rv = 255 - morf
        keypoints = detector.detect(rv)

        # keypoints is a list of keypoint, which include the coordinates (of the centres) of the blobs, and their size

        im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 155), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # End Blob Analysis
        ut.show(Bacground=bckI, Mask=c_mask, Blob=im_with_keypoints)
        key = cv2.waitKey(0)
        bckU = ut.updating_background(morf, gray, bckI, 0.1)
        bckI = bckU.copy()
    nFrame+=1
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
