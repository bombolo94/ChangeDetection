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
                backgroundInit = (s / med)

            background = gray * 0.5 + backgroundInit * (1 - 0.5)
            imgMorphology = cv2.absdiff(gray.astype(np.uint8), background.astype(np.uint8))
            #imgMorphology = ut.denoise(imgMorphology, 3)
            imgMorphology = cv2.threshold(imgMorphology.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)[1]
            #
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            imgMorphology = cv2.dilate(imgMorphology, kernel, iterations = 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
            imgMorphology = cv2.morphologyEx(imgMorphology, cv2.MORPH_CLOSE, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            imgMorphology = cv2.morphologyEx(imgMorphology, cv2.MORPH_OPEN, kernel)

            # Blob Analysis

            params = cv2.SimpleBlobDetector_Params()

            params.minThreshold = 0
            params.maxThreshold = 256

            params.filterByArea = True
            params.minArea = 50
            params.maxArea = imgMorphology.size * (1 / 4)

            params.filterByCircularity = False
            params.minCircularity = 0
            params.maxCircularity = 1

            params.filterByInertia = False
            params.filterByConvexity = False
            params.filterByColor = False

            detector = cv2.SimpleBlobDetector_create(params)

            # Detect blobs
            rv = 255 - imgMorphology
            kp = detector.detect(rv)

            # kp is a list of kp, which include the coordinates (of the centres) of the blobs,
            #  and their size

            imgBlob = cv2.drawKeypoints(gray, kp, np.array([]), (0, 0, 155), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # End Blob Analysis

            # FindCount

            _, cnt, _ = cv2.findContours(imgMorphology, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(frame, cnt, -1, (0, 255, 0), 2)



            for i in range (len(cnt)):
                cn = cnt[i]
                area = cv2.contourArea(cn)
                print(i)
                print(area)

            ut.show(Morf=imgMorphology, Contourns=frame)
           # ut.show(IMG= imgMorphology)
            cv2.waitKey(0)


        nFrame += 1
    else:
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
