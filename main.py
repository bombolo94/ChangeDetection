import numpy as np
import cv2
import utils as ut

cap = cv2.VideoCapture('/home/bombo/Scrivania/video.avi')
nFrame = 0
T = 30

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # this part is background initialisation according to ChangeDetection.pdf
    if nFrame < 100:
        if nFrame == 0:
            prev = gray.copy()
        s = cv2.add(gray.astype(np.int32), prev.astype(np.int32))
        prev = s.copy()

    # this part is background subst and updating according to ChangeDetection.pdf
    else:
        if nFrame == 100:
            bckI = (s/100).astype(np.uint8)

        c_mask = cv2.threshold(cv2.absdiff(gray, bckI), T, 255, cv2.THRESH_BINARY)[1]

        # Morphology
        kernel = np.ones((7, 7), np.uint8)
        opening = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        '''dil = cv2.dilate(c_mask, kernel, iterations=1)
        closing = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))'''
        cv2.imshow("Morphology", opening)

        # Blob Analysis

        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 0
        params.maxThreshold = 256

        params.filterByArea = True
        params.minArea = 500

        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        rv = 255 - opening
        keypoints = detector.detect(rv)

        im_with_keypoints = cv2.drawKeypoints(opening, keypoints, np.array([]), (255, 0, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('BLOB', im_with_keypoints)
        # End Blob Analysis

        # Backround Updating
        bckU = ut.updating_background(c_mask, gray, bckI, 0.1)
        bckI = bckU.copy()
    nFrame = nFrame + 1
    cv2.waitKey(0)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
