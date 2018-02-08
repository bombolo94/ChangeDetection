import numpy as np
import cv2
import utils as ut

camera = cv2.VideoCapture('video.avi')
nFrame = 0
T = 25
med = 100

ret, frame = camera.read()
if ret is True:
    run = True
else:
    run = False

while run:
    ret, frame = camera.read()
    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # this part is background initialisation according to ChangeDetection.pdf
        if nFrame < med:
            if nFrame == 0:
                prev = gray
            s = cv2.add(gray.astype(np.int32), prev.astype(np.int32))
            prev = s

        # this part is background subst and updating according to ChangeDetection.pdf
        else:
            if nFrame == med:
                bckI = (s/med)
            #gray = ut.denoise(gray)
            foreground = ut.getforeground(bckI, gray, 0.5)
            morf = cv2.threshold(foreground, T, 255, cv2.THRESH_BINARY)[1]

            kernel = np.ones((2, 2), np.uint8)
            morf = cv2.dilate(morf, kernel, iterations=1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            morf = cv2.morphologyEx(morf,cv2.MORPH_CLOSE, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morf = cv2.morphologyEx(morf, cv2.MORPH_OPEN, kernel)
            '''dif = cv2.absdiff(gray, bckI)
    
            morf = cv2.threshold(dif, T, 255, cv2.THRESH_BINARY)[1]
    
            # Morphology
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
            morf = cv2.dilate(c_mask, element, iterations=1)
            morf = cv2.erode(morf, element, iterations=1)
            morf = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            morf = cv2.morphologyEx(morf, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))'''
            # Blob Analysis

            params = cv2.SimpleBlobDetector_Params()

            params.minThreshold = 0
            params.maxThreshold = 256

            params.filterByArea = True
            params.minArea= 50
            params.maxArea= (morf.size)*1/4

            params.filterByCircularity = False
            params.minCircularity = 0
            params.maxCircularity = 1

            params.filterByInertia = False
            params.filterByConvexity = False
            params.filterByColor = False


            detector = cv2.SimpleBlobDetector_create(params)

            # Detect blobs
            rv = 255 - morf
            keypoints = detector.detect(rv)

            # keypoints is a list of keypoint, which include the coordinates (of the centres) of the blobs, and their size

            im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 155), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # End Blob Analysis

            # Contorni
            #img = cv2.GaussianBlur(morf, (3, 3), 0)
            # convolute with proper kernels
            laplacian = cv2.Laplacian(morf, cv2.CV_64F)



            # fine contorni

            ut.show(Differenza=foreground, Morf=morf, Blob=im_with_keypoints, Edge= laplacian)
            cv2.waitKey(0)
            '''
            bckU = ut.updating_background(morf, gray, bckI, 0.2)
            bckI = bckU'''

        nFrame += 1
    else:
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
