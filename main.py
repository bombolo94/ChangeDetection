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
        '''c_mask = cv2.threshold(cv2.absdiff(gray, bckI), T, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("Threshold-Substraction", c_mask)
        cv2.imshow("BckI", bckI)
        # Morphology

        kernel = np.array([[0,1,0],[1,1,1],[0,1,1]])
        morf = cv2.dilate(c_mask, kernel, iterations=0)
        kernel = np.array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
        morf = cv2.morphologyEx(morf, cv2.MORPH_CLOSE, cv2.getStructuringElement(kernel))
        morf = cv2.morphologyEx(morf, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
        
        denoised = cv2.GaussianBlur(opening, (5, 5), 0)
        filter = cv2.Laplacian(denoised, cv2.CV_64F)
        cv2.imshow('Laplacian Filter', filter)

        cv2.imshow("Morphology", morf)

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

        im_with_keypoints = cv2.drawKeypoints(morf, keypoints, np.array([]), (0, 0, 155),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('BLOB', im_with_keypoints)
        # End Blob Analysis

        # Backround Updating
        bckU = ut.updating_background(c_mask, gray, bckI, 0.5)
        bckI = bckU.copy()'''
    nFrame = nFrame + 1
    cv2.imshow('Gray', gray)
    key = cv2.waitKey(0)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
